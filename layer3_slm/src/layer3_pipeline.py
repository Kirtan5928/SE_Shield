"""
layer3_slm/src/layer3_pipeline.py
===================================
Layer 3 orchestrator.

Assembles ZeroShotSEClassifier and ExplanationEngine into a single
call-site that honours the Layer 2 → Layer 3 → Layer 4 data contracts.

Input contract (from Layer 2)
------------------------------
  text           : str        — preprocessed message (Layer 1 output)
  risk_score     : int        — 0–100 from Layer 2 Logistic Regression
  layer2_label   : str        — "benign" | "suspicious" (SVM gate output)
  message_id     : str | None — caller-assigned unique ID for this message
  timestamp      : str | None — ISO-8601 timestamp or any sortable string

Output contract (to Layer 4)
------------------------------
  {
    "message_id":  str | None,     # passed through — Layer 4 uses for ordering
    "timestamp":   str | None,     # passed through — Layer 4 uses for windowing
    "label":       str,            # primary top NLI attack sub-type label
    "confidence":  float,          # softmax-normalised entailment for top label
    "top_labels":  [               # top-N overlapping attack types (additive)
      {"label": str, "confidence": float},
      ...
    ],
    "probabilities": dict[str,float], # full distribution, descending, sums to 1.0
    "reason":      str,            # one-sentence dashboard explanation
    "layer2_risk": int,            # Layer 2 LR risk score (passed through)
    "latency_ms":  float,          # NLI inference time (0 for fast-path)
  }

Why message_id and timestamp
------------------------------
Layer 4 is a sliding window engine that aggregates Layer 3 outputs across
a conversation.  It needs to:
  1. Identify each message uniquely (message_id) for deduplication and
     per-message reasoning display in the dashboard.
  2. Order messages chronologically (timestamp) to detect urgency escalation
     and trust-building sequences — the core Layer 4 pattern detection.

Both fields are optional at Layer 3 (None if not provided) and are passed
through unchanged.  Layer 4 may assign its own IDs if the caller does not.

Multi-label rationale
----------------------
SE attacks are taxonomically overlapping.  A message containing both an
account-suspension threat (phishing) AND a login-form redirect
(credential_harvesting) correctly entails BOTH hypotheses.  `top_labels`
surfaces all attack types above meaningful thresholds without breaking the
primary single-label contract Layer 4 consumes.

Fast-path behaviour
--------------------
When Layer 2 SVM labels the message "benign", NLI is skipped entirely.
"benign" is not in ATTACK_LABELS — NLI is sub-type classification only.
The SVM binary decision is the correct gate for benign/attack.
"""

from __future__ import annotations

import logging
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAYER3_ROOT = os.path.dirname(_HERE)
if _LAYER3_ROOT not in sys.path:
    sys.path.insert(0, _LAYER3_ROOT)

from src.nli_classifier import ZeroShotSEClassifier
from src.explainer import ExplanationEngine, SESignalExtractor

logger = logging.getLogger(__name__)

# Import defaults from config — callers can override via constructor
try:
    _LAYER3_ROOT_CFG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _LAYER3_ROOT_CFG)
    from config_layer3 import (
        MIN_SUBTYPE_CONFIDENCE        as _DEFAULT_MIN_CONF,
        SIGNAL_GATED_BENIGN_THRESHOLD as _DEFAULT_SIG_BENIGN,
        TOP_LABELS_MAX_COUNT          as _DEFAULT_TOP_N,
        TOP_LABELS_MIN_CONF           as _DEFAULT_TOP_MIN,
        TOP_LABELS_REL_FLOOR          as _DEFAULT_TOP_REL,
    )
except ImportError:
    _DEFAULT_MIN_CONF   = 0.25
    _DEFAULT_SIG_BENIGN = 0.50
    _DEFAULT_TOP_N      = 3
    _DEFAULT_TOP_MIN    = 0.15
    _DEFAULT_TOP_REL    = 0.35


class Layer3Pipeline:
    """
    Full Layer 3 inference pipeline.

    Parameters
    ----------
    model_name              : HuggingFace model identifier.
    labels                  : Ordered list of attack-type label strings.
    hypothesis_templates    : {label: hypothesis string} mapping.
    layer2_threshold        : risk_score at-or-below which NLI is skipped.
    max_length              : Tokeniser truncation length.
    device                  : "cpu" | "cuda" | None (auto-detect).
    min_subtype_confidence  : Below this, top label falls back to "phishing".
    top_labels_max_count    : Max entries in top_labels list (default 3).
    top_labels_min_conf     : Absolute floor to appear in top_labels.
    top_labels_rel_floor    : Must be ≥ this fraction of top label's score.
    """

    def __init__(
        self,
        model_name: str,
        labels: list[str],
        hypothesis_templates: dict[str, str],
        layer2_threshold: int = 50,
        max_length: int = 256,
        device: str | None = None,
        min_subtype_confidence: float | None = None,
        signal_gated_benign_threshold: float | None = None,
        top_labels_max_count: int | None = None,
        top_labels_min_conf: float | None = None,
        top_labels_rel_floor: float | None = None,
    ) -> None:
        self.layer2_threshold             = layer2_threshold
        self.min_subtype_confidence       = min_subtype_confidence       if min_subtype_confidence       is not None else _DEFAULT_MIN_CONF
        self.signal_gated_benign_threshold = signal_gated_benign_threshold if signal_gated_benign_threshold is not None else _DEFAULT_SIG_BENIGN
        self.top_labels_max_count         = top_labels_max_count         if top_labels_max_count         is not None else _DEFAULT_TOP_N
        self.top_labels_min_conf          = top_labels_min_conf          if top_labels_min_conf          is not None else _DEFAULT_TOP_MIN
        self.top_labels_rel_floor         = top_labels_rel_floor         if top_labels_rel_floor         is not None else _DEFAULT_TOP_REL

        self.classifier = ZeroShotSEClassifier(
            model_name=model_name,
            labels=labels,
            hypothesis_templates=hypothesis_templates,
            max_length=max_length,
            device=device,
        )
        self.explainer  = ExplanationEngine()
        self._signals   = SESignalExtractor()

        logger.info(
            "Layer3Pipeline ready. "
            "Threshold=%d  min_subtype_conf=%.2f  sig_benign_thresh=%.2f  "
            "top_labels(max=%d, min=%.2f, rel=%.2f)  labels=%s",
            layer2_threshold,
            self.min_subtype_confidence,
            self.signal_gated_benign_threshold,
            self.top_labels_max_count,
            self.top_labels_min_conf,
            self.top_labels_rel_floor,
            labels,
        )

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        text: str,
        layer2_risk_score: int = 100,
        layer2_label: str = "suspicious",
        message_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        """
        Classify one message.

        Parameters
        ----------
        text              : Preprocessed message text (Layer 1 output).
        layer2_risk_score : LR risk score 0–100 from Layer 2.
        layer2_label      : SVM decision — "benign" or "suspicious".
                            "benign" triggers fast-path; NLI is skipped.
        message_id        : Caller-assigned unique message identifier.
                            Passed through to output for Layer 4 ordering.
                            If None, Layer 4 should assign its own ID.
        timestamp         : ISO-8601 string or any sortable timestamp.
                            Passed through to output for Layer 4 windowing.
                            Layer 4 uses this for temporal pattern detection.

        Returns
        -------
        Layer 3 output contract dict.
        """
        # ── Fast-path: Layer 2 SVM classified this message as benign ────────
        if layer2_label == "benign" or layer2_risk_score <= self.layer2_threshold:
            logger.debug(
                "Fast-path: svm_label=%s risk_score=%d — skipping NLI.",
                layer2_label, layer2_risk_score,
            )
            return self._benign_passthrough(layer2_risk_score, message_id, timestamp)

        # ── NLI classification ───────────────────────────────────────────────
        clf = self.classifier.classify(text)

        # ── Confidence fallback ──────────────────────────────────────────────
        if clf["confidence"] < self.min_subtype_confidence:
            clf = {**clf, "label": "phishing"}

        # ── Build top_labels ─────────────────────────────────────────────────
        top_labels = self._build_top_labels(clf["probabilities"])

        reason = self.explainer.explain(text, clf)

        return {
            "message_id":    message_id,
            "timestamp":     timestamp,
            "label":         clf["label"],
            "confidence":    clf["confidence"],
            "top_labels":    top_labels,
            "probabilities": clf["probabilities"],
            "reason":        reason,
            "layer2_risk":   layer2_risk_score,
            "latency_ms":    clf["latency_ms"],
        }

    def run_batch(self, items: list[dict]) -> list[dict]:
        """
        Classify a list of messages.

        Each item dict keys:
          text          : str   (required)
          risk_score    : int   (optional, default 100)
          layer2_label  : str   (optional, default "suspicious")
          message_id    : str   (optional, default None)
          timestamp     : str   (optional, default None)

        Returns list of output contract dicts, same order as input.

        Example
        -------
        results = pipe.run_batch([
            {
                "text":         "Urgent: verify your account",
                "risk_score":   82,
                "layer2_label": "suspicious",
                "message_id":   "msg_001",
                "timestamp":    "2026-05-08T09:15:00Z",
            },
            {
                "text":         "See you at 3pm",
                "risk_score":   5,
                "layer2_label": "benign",
                "message_id":   "msg_002",
                "timestamp":    "2026-05-08T09:16:00Z",
            },
        ])
        """
        results = []
        for i, item in enumerate(items):
            result = self.run(
                text=item["text"],
                layer2_risk_score=item.get("risk_score", 100),
                layer2_label=item.get("layer2_label", "suspicious"),
                message_id=item.get("message_id", None),
                timestamp=item.get("timestamp", None),
            )
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info("  processed %d / %d messages", i + 1, len(items))
        return results

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _build_top_labels(self, probabilities: dict[str, float]) -> list[dict]:
        """
        Build the top_labels list from the full probability distribution.

        Inclusion rules (both must pass):
          1. confidence >= top_labels_min_conf        (absolute floor)
          2. confidence >= top_conf * top_labels_rel_floor  (relative floor)

        "benign" is always excluded from top_labels — it has no sub-type
        meaning for the attack taxonomy and its presence/absence is carried
        by the primary `label` field.

        Returns list of {"label": str, "confidence": float}, descending,
        length between 1 and top_labels_max_count.
        """
        top_conf = max(probabilities.values())
        rel_threshold = top_conf * self.top_labels_rel_floor

        qualified = [
            {"label": lbl, "confidence": conf}
            for lbl, conf in probabilities.items()
            if lbl != "benign"
            and conf >= self.top_labels_min_conf
            and conf >= rel_threshold
        ]

        # Already sorted descending (probabilities dict is sorted on creation)
        return qualified[: self.top_labels_max_count]

    @staticmethod
    def _benign_passthrough(
        risk_score: int,
        message_id: str | None = None,
        timestamp:  str | None = None,
    ) -> dict:
        """
        Output contract for messages cleared by Layer 2 SVM (fast-path).

        Confidence = 1 − (risk_score / 100) so Layer 4 sees a calibrated
        benign confidence even for gated messages.
        top_labels = [] — no NLI ran, no attack sub-type evaluated.
        message_id and timestamp are passed through unchanged for Layer 4.
        """
        return {
            "message_id":    message_id,
            "timestamp":     timestamp,
            "label":         "benign",
            "confidence":    round(1.0 - risk_score / 100.0, 4),
            "top_labels":    [],
            "probabilities": {"benign": 1.0},
            "reason":        (
                "Layer 2 SVM classified this message as benign — "
                "no NLI inference required."
            ),
            "layer2_risk":   risk_score,
            "latency_ms":    0.0,
        }