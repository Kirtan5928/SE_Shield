"""
layer4/hybrid_pipeline.py
===========================
Full hybrid SE detection pipeline.

Wires all layers in the final confirmed architecture:

  Raw text
    → Layer 1 : adversarial normalisation
    → Layer 2 : SVM (binary gate) + LR (risk_score)
    → Signal-gated promotion: SVM-benign messages are promoted to NLI when
                (a) ≥2 SE signal categories match, OR
                (b) the credential category matches (solo-qualifying), OR
                (c) LR risk_score ≥ 50 while SVM says benign (model disagreement)
    → Layer 4a: Risk Counter (conversation context, SVM override)
    → Layer 3 : NLI sub-type classification
    → Layer 4b: Semantic Window (pattern detection, entity risk)
    → Output  : dashboard-ready conversation assessment

Usage
-----
  from layer4.hybrid_pipeline import HybridPipeline

  pipe = HybridPipeline()

  # Process one message at a time (streaming, real-time)
  result = pipe.process(
      text            = "Urgent: verify your account now.",
      conversation_id = "conv_42",
      message_id      = "conv_42_msg_003",
      timestamp       = "2026-05-08T09:15:00Z",
  )
  print(result["alert_level"])   # "HIGH"
  print(result["attack_pattern"]) # "authority_then_credential"

  # Process a full conversation at once (batch / test mode)
  results = pipe.process_conversation(conversation)

Phase 2 changes
---------------
Only change: pass real conversation_id from your data source.
  email → thread_id or hash(sender+recipient)
  chat  → session_id or room_id
  SMS   → hash(from_number + to_number)

All layer internals unchanged.

SIGNAL-GATED PROMOTION (Problem 1 + Problem 2 fix)
---------------------------------------------------
The SVM gate is trained on standard-English phishing corpora. Two failure
modes were observed:

  1. Non-standard / Indian English ("reserver bank of India", "requesting
     you for your cooperation") falls outside the TF-IDF vocabulary →
     SVM calls it benign → NLI never runs.
  2. Conversational credential requests between "colleagues" ("send me
     your username and password right now") read as informal chat to the
     SVM → benign → NLI never runs — even though this is the exploit
     message of a trust-building attack.

Promotion rules (any one suffices, checked only when SVM says benign):

  RULE A — ≥ SIGNAL_PROMOTE_MIN_CATEGORIES distinct signal categories
           (e.g. credential + authority: the Rajesh RBI message).
  RULE B — the credential category alone. Direct mention of password /
           username / login / OTP / PIN is the single highest-precision
           SE signal; it must never be gated out by the SVM.
  RULE C — LR risk_score ≥ SIGNAL_PROMOTE_LR_THRESHOLD. SVM and LR are
           trained on the same features; when the LR leans attack while
           the SVM says benign, the gate is uncertain — delegate the
           decision to NLI.

Safety properties:
  * Promotion ONLY — a message is never demoted to benign by this check,
    so recall can only increase. This is the inverse of the (disabled)
    SIGNAL_GATED_BENIGN_THRESHOLD mechanism that destroyed recall.
  * Promoted messages also feed "suspicious" into Layer 4a with a
    risk-score floor, so the conversation accumulator rises consistently
    with the per-message verdict instead of decaying.
  * Cost: one regex pass (~microseconds) + NLI inference (~150ms) on
    promoted messages. Acceptable for a conversation-level pipeline.
  * Precision tradeoff: benign messages that genuinely discuss passwords
    or score LR ≥ 50 will now receive NLI labels (possibly mild attack
    labels). Consistent with the project's accepted FN=0-first policy.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent        # layer4/
_PROJECT_ROOT = _HERE.parent                           # hybrid_se/
_LAYER3_DIR   = _PROJECT_ROOT / "layer3_slm"

# Force layer3_slm to sys.path[0] regardless of existing state.
# If layer3_slm is already in sys.path (inserted by test runner or caller),
# it may be at a position AFTER hybrid_se/src, causing `from src.layer3_pipeline`
# to resolve against hybrid_se/src/ instead of layer3_slm/src/.
# Remove and re-insert at 0 to guarantee correct resolution order.
_l3_str = str(_LAYER3_DIR)
if _l3_str in sys.path:
    sys.path.remove(_l3_str)
sys.path.insert(0, _l3_str)

# Project root goes in second position (needed for layer4 imports).
_pr_str = str(_PROJECT_ROOT)
if _pr_str not in sys.path:
    sys.path.insert(1, _pr_str)

from layer4.layer4a_risk_counter    import RiskCounter
from layer4.layer4b_semantic_window import SemanticWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal-gated promotion config
# ---------------------------------------------------------------------------

# RULE A — minimum number of DISTINCT signal categories (urgency /
# credential / authority / financial / threat) that must match before an
# SVM-benign message is promoted to NLI.
SIGNAL_PROMOTE_MIN_CATEGORIES = 2

# RULE B — categories that qualify on their own (no second category needed).
# "credential" is solo-qualifying: a message mentioning password / username /
# login / OTP / PIN must always be inspected by NLI, regardless of SVM.
SIGNAL_PROMOTE_SOLO_CATEGORIES: set[str] = {"credential"}

# RULE C — LR/SVM disagreement. If the SVM says benign but the LR risk
# score is at or above this value, the gate is uncertain → promote to NLI.
SIGNAL_PROMOTE_LR_THRESHOLD = 60

# Risk-score floor applied when a message is promoted. Ensures the L4a
# accumulator and the L4b confidence weighting receive a meaningful weight
# even when the LR (also trained on standard English) was lukewarm.
SIGNAL_PROMOTE_RISK_FLOOR = 40


# ---------------------------------------------------------------------------
# Minimal Layer 1 normaliser (reuse from integrate_layers if available)
# ---------------------------------------------------------------------------

class _Layer1:
    _SUBS = str.maketrans({
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "@": "a", "$": "s",
        "\u0430": "a", "\u0435": "e", "\u043e": "o",
    })

    def normalise(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"%[0-9A-Fa-f]{2}",
                      lambda m: bytes.fromhex(m.group(0)[1:]).decode("latin-1", errors="replace"),
                      text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.translate(self._SUBS)


# ---------------------------------------------------------------------------
# Hybrid Pipeline
# ---------------------------------------------------------------------------

class HybridPipeline:
    """
    Full hybrid SE detection pipeline (L1 → L2 → L4a → L3 → L4b).

    Parameters
    ----------
    tfidf_path / svm_path / lr_path : Layer 2 model paths.
    layer3_model_name               : HuggingFace model ID.
    window_size                     : Sliding window depth (default 10).
    suspicious_threshold            : L4a threshold for SVM override (default 3.0).
    device                          : "cpu" | "cuda" | None (auto).
    signal_promote_min_categories   : RULE A threshold (default 2).
    signal_promote_lr_threshold     : RULE C threshold (default 50).
    """

    def __init__(
        self,
        tfidf_path:         str | Path | None = None,
        svm_path:           str | Path | None = None,
        lr_path:            str | Path | None = None,
        layer3_model_name:  str = "cross-encoder/nli-deberta-v3-small",
        window_size:        int = 10,
        suspicious_threshold: float = 3.0,
        device:             Optional[str] = None,
        signal_promote_min_categories: int = SIGNAL_PROMOTE_MIN_CATEGORIES,
        signal_promote_lr_threshold:   int = SIGNAL_PROMOTE_LR_THRESHOLD,
    ) -> None:
        import joblib
        # Re-assert layer3_slm at sys.path[0] immediately before local imports.
        # Defends against external code (e.g. server identity import block)
        # inserting paths that push layer3_slm away from position 0 between
        # module load time and __init__ call time.
        _l3 = str(_LAYER3_DIR)
        if _l3 in sys.path:
            sys.path.remove(_l3)
        sys.path.insert(0, _l3)

        from config_layer3 import (
            ATTACK_LABELS, HYPOTHESIS_TEMPLATES,
            LAYER2_THRESHOLD, MAX_LENGTH,
            URGENCY_SIGNALS, CREDENTIAL_SIGNALS, AUTHORITY_SIGNALS,
            FINANCIAL_SIGNALS, THREAT_SIGNALS,
        )
        from src.layer3_pipeline import Layer3Pipeline

        _models = _PROJECT_ROOT / "models"
        tfidf_path = tfidf_path or _models / "tfidf_vectorizer.pkl"
        svm_path   = svm_path   or _models / "stage1a_svm_final.pkl"
        lr_path    = lr_path    or _models / "stage1b_lr_final.pkl"

        logger.info("Loading Layer 2 models …")
        self._tfidf = joblib.load(tfidf_path)
        self._svm   = joblib.load(svm_path)
        self._lr    = joblib.load(lr_path)
        logger.info("Layer 2 ready.")

        self._l1 = _Layer1()

        # Pre-compile signal regexes once (used by signal-gated promotion).
        # Categories mirror the ExplanationEngine lists in config_layer3.
        self.signal_promote_min_categories = signal_promote_min_categories
        self.signal_promote_lr_threshold   = signal_promote_lr_threshold
        self._signal_categories: dict[str, list[re.Pattern]] = {
            "urgency":    [re.compile(p, re.IGNORECASE) for p in URGENCY_SIGNALS],
            "credential": [re.compile(p, re.IGNORECASE) for p in CREDENTIAL_SIGNALS],
            "authority":  [re.compile(p, re.IGNORECASE) for p in AUTHORITY_SIGNALS],
            "financial":  [re.compile(p, re.IGNORECASE) for p in FINANCIAL_SIGNALS],
            "threat":     [re.compile(p, re.IGNORECASE) for p in THREAT_SIGNALS],
        }

        self._l3 = Layer3Pipeline(
            model_name           = layer3_model_name,
            labels               = ATTACK_LABELS,
            hypothesis_templates = HYPOTHESIS_TEMPLATES,
            layer2_threshold     = LAYER2_THRESHOLD,
            max_length           = MAX_LENGTH,
            device               = device,
        )

        self._l4a = RiskCounter(
            window_size          = window_size,
            suspicious_threshold = suspicious_threshold,
        )
        self._l4b = SemanticWindow(window_size=window_size)

        logger.info("HybridPipeline ready.")

    # ------------------------------------------------------------------ #
    # Signal-gated promotion                                                #
    # ------------------------------------------------------------------ #

    def _matched_signal_categories(self, text: str) -> list[str]:
        """
        Return the list of distinct SE signal categories present in `text`.
        Used to promote SVM-benign messages to NLI when SE vocabulary is
        present despite the SVM verdict (non-standard English, OOV phrasing,
        conversational credential requests).
        """
        matched = []
        for category, patterns in self._signal_categories.items():
            if any(p.search(text) for p in patterns):
                matched.append(category)
        return matched

    def _should_promote(
        self, matched_signals: list[str], risk_score: int
    ) -> tuple[bool, str]:
        """
        Decide whether an SVM-benign message should be promoted to NLI.

        Returns (promote, rule) where rule is a short tag for logging /
        dashboard trail: "multi_signal" | "credential_solo" | "lr_disagree".
        """
        # RULE B — solo-qualifying category (credential)
        if any(c in SIGNAL_PROMOTE_SOLO_CATEGORIES for c in matched_signals):
            return True, "credential_solo"

        # RULE A — multiple distinct categories
        if len(matched_signals) >= self.signal_promote_min_categories:
            return True, "multi_signal"

        # RULE C — SVM/LR disagreement
        if risk_score >= self.signal_promote_lr_threshold and matched_signals:
            return True, "lr_disagree"

        return False, ""

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def process(
        self,
        text:            str,
        conversation_id: str,
        message_id:      str | None = None,
        timestamp:       str | None = None,
    ) -> dict:
        """
        Process one message through the full pipeline.

        Returns the Layer 4b dashboard output contract, extended with
        the per-message Layer 3 result under key "last_message".

        Parameters
        ----------
        text            : Raw message text.
        conversation_id : Unique conversation ID.
                          Phase 1: synthetic ("conv_42")
                          Phase 2: thread_id / session_id / hash(from+to)
        message_id      : Optional unique message ID.
        timestamp       : Optional ISO-8601 timestamp.
        """
        # ── L1: normalise ────────────────────────────────────────────────
        clean = self._l1.normalise(text)

        # ── L2: SVM + LR ─────────────────────────────────────────────────
        vec        = self._tfidf.transform([clean])
        svm_pred   = int(self._svm.predict(vec)[0])
        svm_label  = "suspicious" if svm_pred == 1 else "benign"
        risk_score = int(self._lr.predict_proba(vec)[0][1] * 100)

        # ── Signal-gated promotion (Problems 1 & 2 fix) ──────────────────
        # Checked only when SVM says benign. PROMOTION ONLY — never demotes.
        signal_promote   = False
        promote_rule     = ""
        matched_signals: list[str] = []
        if svm_label == "benign":
            matched_signals = self._matched_signal_categories(clean)
            signal_promote, promote_rule = self._should_promote(
                matched_signals, risk_score
            )
            if signal_promote:
                logger.info(
                    "Signal-gated promotion [%s]: conv=%s categories=%s lr=%d "
                    "(SVM said benign — forcing NLI)",
                    promote_rule, conversation_id, matched_signals, risk_score,
                )

        # Label and risk-score the rest of the pipeline should see.
        # Promoted messages get a risk floor so the L4a accumulator and the
        # L4b confidence weighting don't treat the message as near-zero risk.
        l4a_label      = "suspicious" if signal_promote else svm_label
        effective_risk = (
            max(risk_score, SIGNAL_PROMOTE_RISK_FLOOR)
            if signal_promote else risk_score
        )

        # ── L4a: Risk Counter ────────────────────────────────────────────
        ctx = self._l4a.update(
            conversation_id = conversation_id,
            svm_label       = l4a_label,
            risk_score      = effective_risk,
        )

        # Apply SVM override: if conversation is suspicious but this
        # message looked benign, force Layer 3 to run NLI anyway.
        # Signal promotion also forces NLI.
        effective_svm = (
            "suspicious"
            if (ctx["override_svm"] or signal_promote)
            else svm_label
        )

        # Apply dynamic confidence threshold from L4a context.
        self._l3.min_subtype_confidence = ctx["recommended_min_conf"]

        # ── L3: NLI classification ───────────────────────────────────────
        l3_result = self._l3.run(
            text              = clean,
            layer2_risk_score = effective_risk,
            layer2_label      = effective_svm,
            message_id        = message_id,
            timestamp         = timestamp,
        )

        # Surface promotion in the per-message reason (dashboard trail).
        if signal_promote and l3_result.get("label") != "benign":
            tag = ", ".join(matched_signals) if matched_signals else promote_rule
            l3_result["reason"] = (
                f"[signal-promoted ({promote_rule}): {tag}] "
                + l3_result.get("reason", "")
            )

        # ── L4b: Semantic Window ─────────────────────────────────────────
        assessment = self._l4b.update(conversation_id, l3_result)

        # Combine: return conversation assessment + per-message detail
        return {
            **assessment,
            "last_message": l3_result,
            "l4a_context":  {
                **ctx,
                "signal_promoted":   signal_promote,
                "promote_rule":      promote_rule,
                "matched_signals":   matched_signals,
                "svm_raw_label":     svm_label,
                "lr_raw_risk_score": risk_score,
            },
        }

    def process_conversation(self, messages: list[dict]) -> list[dict]:
        """
        Process a full conversation sequentially.

        Each message dict:
          text            : str  (required)
          conversation_id : str  (required)
          message_id      : str  (optional)
          timestamp       : str  (optional)

        Returns list of per-message pipeline outputs (cumulative assessment
        grows with each message).

        Example
        -------
        conv = [
            {"text": "Hi I'm from IT",            "conversation_id": "conv_42",
             "message_id": "m1", "timestamp": "2026-05-08T09:00:00Z"},
            {"text": "Need your login to fix it",  "conversation_id": "conv_42",
             "message_id": "m2", "timestamp": "2026-05-08T09:05:00Z"},
        ]
        results = pipe.process_conversation(conv)
        print(results[-1]["alert_level"])   # "HIGH"
        """
        results = []
        for msg in messages:
            result = self.process(
                text            = msg["text"],
                conversation_id = msg["conversation_id"],
                message_id      = msg.get("message_id"),
                timestamp       = msg.get("timestamp"),
            )
            results.append(result)
        return results

    def reset_conversation(self, conversation_id: str) -> None:
        """Reset both L4a and L4b state for a resolved conversation."""
        self._l4a.reset(conversation_id)
        self._l4b.reset(conversation_id)

    def active_conversations(self) -> dict:
        """Return summary of all active conversation windows."""
        return {
            "layer4a": self._l4a.list_active(),
            "layer4b": self._l4b.list_active(),
        }