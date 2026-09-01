"""
src/integrate_layers.py
========================
End-to-end Layer 2 → Layer 3 integration.

Wires the existing pickled Layer 2 models (TF-IDF + SVM + LR) into the
Layer 3 pipeline.  This is the full hybrid pipeline as it will run in
production and in the Streamlit dashboard.

Data flow
---------
  Raw text
    → Layer 1: adversarial normalisation  (existing — loaded from preprocess.py)
    → Layer 2 TF-IDF:  vectorise
    → Layer 2 SVM:     binary gate (benign → EXIT, suspicious → continue)
    → Layer 2 LR:      P(attack) → risk_score 0–100
    → risk_score > LAYER2_THRESHOLD?
        → Layer 3 NLI: ZeroShotSEClassifier
        → ExplanationEngine → reason
    → Output contract (ready for Layer 4)

Usage
-----
  from src.integrate_layers import HybridSEPipeline
  pipe = HybridSEPipeline()
  result = pipe.run("Urgent: verify your account immediately.")
  print(result)

  # Or process a batch:
  results = pipe.run_batch(["msg1", "msg2", ...])
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent       # src/
_PROJECT_ROOT = _HERE.parent                          # hybrid_se/
_LAYER3_DIR   = _PROJECT_ROOT / "layer3_slm"

for p in [str(_LAYER3_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config_layer3 import (
    ATTACK_LABELS,
    HYPOTHESIS_TEMPLATES,
    LAYER2_THRESHOLD,
    MAX_LENGTH,
    MODEL_NAME,
)
from src.layer3_pipeline import Layer3Pipeline

logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
_MODELS_DIR      = _PROJECT_ROOT / "models"
_TFIDF_PATH      = _MODELS_DIR / "tfidf_vectorizer.pkl"
_SVM_PATH        = _MODELS_DIR / "stage1a_svm_final.pkl"
_LR_PATH         = _MODELS_DIR / "stage1b_lr_final.pkl"


# ---------------------------------------------------------------------------
# Minimal Layer 1 normaliser
# (mirrors what the full adversarial preprocessor does — replace with the
#  real Layer 1 import once it is importable as a module)
# ---------------------------------------------------------------------------

class _Layer1Normaliser:
    """
    Lightweight adversarial text normaliser.
    Handles the most common obfuscation patterns seen in SE attacks.
    Replace with `from src.adversarial_prep import normalise` if you expose
    the full Layer 1 as a function.
    """

    # Common l33tspeak / homoglyph substitutions
    _SUBS = str.maketrans({
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "@": "a", "$": "s",
        "\u0430": "a",   # Cyrillic а → Latin a
        "\u0435": "e",   # Cyrillic е → Latin e
        "\u043e": "o",   # Cyrillic о → Latin o
    })

    def normalise(self, text: str) -> str:
        # Unicode normalise (NFC)
        text = unicodedata.normalize("NFC", text)
        # Decode percent-encoded URLs partially (e.g. %20 → space)
        text = re.sub(r"%[0-9A-Fa-f]{2}", lambda m: bytes.fromhex(m.group(0)[1:]).decode("latin-1", errors="replace"), text)
        # Collapse whitespace abuse
        text = re.sub(r"\s+", " ", text).strip()
        # Apply homoglyph substitution
        text = text.translate(self._SUBS)
        return text


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------

class HybridSEPipeline:
    """
    Full Layer 1 → Layer 2 → Layer 3 pipeline.

    Parameters
    ----------
    tfidf_path   : Path to pickled TF-IDF vectoriser.
    svm_path     : Path to pickled SVM gate model.
    lr_path      : Path to pickled LR risk scorer.
    layer3_*     : Forwarded to Layer3Pipeline.
    verbose      : If True, log Layer 2 intermediate results.

    Example
    -------
    pipe = HybridSEPipeline()
    result = pipe.run("Your account will be suspended — verify now.")
    # → {
    #     "label": "phishing",
    #     "confidence": 0.87,
    #     "probabilities": {...},
    #     "reason": "Message creates urgency ... — consistent with phishing.",
    #     "layer2_risk": 83,
    #     "latency_ms": 154.2,
    #     "layer2_label": "suspicious",
    #     "layer2_svm_hit": True,
    #     "total_latency_ms": 156.1,
    #   }
    """

    def __init__(
        self,
        tfidf_path:  Path | str = _TFIDF_PATH,
        svm_path:    Path | str = _SVM_PATH,
        lr_path:     Path | str = _LR_PATH,
        model_name:  str = MODEL_NAME,
        labels:      list[str] | None = None,
        templates:   dict[str, str] | None = None,
        threshold:   int = LAYER2_THRESHOLD,
        max_length:  int = MAX_LENGTH,
        device:      Optional[str] = None,
        verbose:     bool = False,
    ) -> None:
        self.verbose   = verbose
        self.threshold = threshold

        # ── Layer 1 ──────────────────────────────────────────────────────────
        self._l1 = _Layer1Normaliser()

        # ── Layer 2 ──────────────────────────────────────────────────────────
        logger.info("Loading Layer 2 models …")
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")

        self._tfidf = joblib.load(tfidf_path)
        self._svm   = joblib.load(svm_path)
        self._lr    = joblib.load(lr_path)
        logger.info("Layer 2 models loaded.")

        # ── Layer 3 ──────────────────────────────────────────────────────────
        self._l3 = Layer3Pipeline(
            model_name=model_name,
            labels=labels or ATTACK_LABELS,
            hypothesis_templates=templates or HYPOTHESIS_TEMPLATES,
            layer2_threshold=threshold,
            max_length=max_length,
            device=device,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def run(self, raw_text: str) -> dict:
        """
        Process one raw message through the full pipeline.

        Returns
        -------
        Layer 3 output contract dict, extended with:
          layer2_label     : "benign" | "suspicious"  (SVM output)
          layer2_svm_hit   : bool   (True if SVM flagged it suspicious)
          total_latency_ms : float  (L1 + L2 + L3 wall time)
        """
        t_start = time.perf_counter()

        # ── Layer 1: normalise ──────────────────────────────────────────────
        clean = self._l1.normalise(raw_text)

        # ── Layer 2: TF-IDF + SVM + LR ──────────────────────────────────────
        vec         = self._tfidf.transform([clean])
        svm_pred    = int(self._svm.predict(vec)[0])
        svm_label   = "suspicious" if svm_pred == 1 else "benign"
        prob_attack = float(self._lr.predict_proba(vec)[0][1])
        risk_score  = int(prob_attack * 100)

        if self.verbose:
            logger.info(
                "Layer 2: svm=%s  risk_score=%d  text='%s…'",
                svm_label, risk_score, clean[:60],
            )

        # ── Layer 3: NLI ─────────────────────────────────────────────────────
        l3_result = self._l3.run(
            text=clean,
            layer2_risk_score=risk_score,
            layer2_label=svm_label,
        )

        total_ms = (time.perf_counter() - t_start) * 1_000

        return {
            **l3_result,
            "layer2_label":     svm_label,
            "layer2_svm_hit":   svm_pred == 1,
            "total_latency_ms": round(total_ms, 1),
        }

    def run_batch(self, texts: list[str], log_every: int = 100) -> list[dict]:
        """Process a list of raw messages. Returns one result dict per message."""
        results = []
        for i, text in enumerate(texts):
            results.append(self.run(text))
            if log_every and (i + 1) % log_every == 0:
                logger.info("  processed %d / %d", i + 1, len(texts))
        return results


# ---------------------------------------------------------------------------
# Quick integration smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    EXAMPLES = [
        "Urgent: Your Microsoft account password must be changed immediately. Click here.",
        "Hi John, this is the CEO. Can you wire $25,000 to our new supplier today?",
        "Hey, are you free for coffee tomorrow afternoon?",
        "Your Netflix subscription has expired. Confirm your payment details to continue.",
        "The team lunch is at 12:30 on Friday. Room has been booked.",
    ]

    print("\nLoading full hybrid pipeline …")
    try:
        pipe = HybridSEPipeline(verbose=True)
    except FileNotFoundError as e:
        print(f"\nModel files not found: {e}")
        print("Make sure you are running from the project root (hybrid_se/).")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  FULL PIPELINE (Layer 1 → Layer 2 → Layer 3) SMOKE TEST")
    print("=" * 70)

    for text in EXAMPLES:
        r = pipe.run(text)
        print(f"\n  Input      : {text[:70]}")
        print(f"  L2 label   : {r['layer2_label']}  risk={r['layer2_risk']}")
        print(f"  L3 label   : {r['label'].upper()}  conf={r['confidence']:.3f}")
        print(f"  Reason     : {r['reason']}")
        print(f"  Total ms   : {r['total_latency_ms']}")

    print("\n" + "=" * 70)