"""
layer3_slm/src/nli_classifier.py
==================================
Zero-shot SE attack classifier using NLI entailment scoring.

Architecture
------------
cross-encoder/nli-deberta-v3-small takes (premise, hypothesis) pairs
and emits three logits per pair.  The label order is model-specific and
is read from the model's own config.id2label at load time — never hardcoded.

For cross-encoder/nli-deberta-v3-small the label order is:
  {0: "contradiction", 1: "entailment", 2: "neutral"}
  → entailment is at index 1.

PREVIOUS BUG: _ENTAILMENT_IDX was hardcoded to 2, which selected the
NEUTRAL logit instead of the entailment logit.  This caused all labels
to receive similar neutral scores (0.20–0.50), making argmax essentially
random and causing baiting/spear_phishing to win by default.

For a single message we:
  1. Construct N (premise=message, hypothesis=label_template) pairs.
  2. Stack all N pairs into a single tokenised batch.
  3. Run ONE forward pass → shape (N, 3) logits.
  4. Slice the ENTAILMENT column → shape (N,) entailment logits.
  5. Softmax over those N logits → calibrated probability distribution.
  6. argmax → predicted label + confidence.

Batching gives ~6× speedup vs N separate forward passes (~150ms total
vs ~1,200ms for 8 labels sequentially on Mac CPU).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ZeroShotSEClassifier:
    """
    Zero-shot social engineering classifier using NLI.

    The entailment column index is determined at load time by reading the
    model's config.id2label — not hardcoded — so this class works correctly
    with any cross-encoder NLI model regardless of label order.

    Parameters
    ----------
    model_name           : HuggingFace model identifier.
    labels               : Ordered list of attack-type label strings.
    hypothesis_templates : Mapping {label: natural-language hypothesis string}.
    max_length           : Tokeniser truncation length (default 256).
    device               : "cpu" | "cuda" — auto-detected if None.
    """

    def __init__(
        self,
        model_name: str,
        labels: list[str],
        hypothesis_templates: dict[str, str],
        max_length: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.labels = labels
        self.hypothesis_templates = hypothesis_templates
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading NLI model '%s' on %s …", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # ── Detect entailment column index from the model's own config ───────
        # Different NLI models use different label orders.
        # cross-encoder/nli-deberta-v3-small: {0: contradiction, 1: entailment, 2: neutral}
        # Some BERT-MNLI models use:          {0: entailment,    1: neutral,    2: contradiction}
        # Reading id2label handles both correctly without hardcoding.
        id2label: dict = self.model.config.id2label          # {int_or_str: label_str}
        label2id = {v.lower(): int(k) for k, v in id2label.items()}

        if "entailment" not in label2id:
            logger.warning(
                "Could not find 'entailment' in model label map %s — "
                "defaulting to index 1.  Verify manually.", label2id
            )
        self._entailment_idx: int = label2id.get("entailment", 1)

        logger.info(
            "Model ready. Label map: %s  →  using entailment_idx=%d",
            {int(k): v for k, v in id2label.items()},
            self._entailment_idx,
        )
        logger.info("Attack labels (%d): %s", len(labels), ", ".join(labels))

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def classify(self, text: str) -> dict:
        """
        Classify a single message.

        Returns
        -------
        dict:
          label         : str   — top-scoring label
          confidence    : float — softmax-normalised entailment score for top label
          probabilities : dict  — {label: float} sorted descending, sums to 1.0
          raw_scores    : dict  — {label: float} pre-softmax entailment logits
          latency_ms    : float — wall-clock inference time in milliseconds
        """
        t0  = time.perf_counter()
        raw = self._batched_forward(text)
        probs = self._softmax(raw)
        top = max(probs, key=probs.__getitem__)
        ms  = (time.perf_counter() - t0) * 1_000

        return {
            "label":         top,
            "confidence":    round(float(probs[top]), 4),
            "probabilities": {
                k: round(float(v), 4)
                for k, v in sorted(probs.items(), key=lambda x: -x[1])
            },
            "raw_scores":    {k: round(float(v), 4) for k, v in raw.items()},
            "latency_ms":    round(ms, 1),
        }

    def classify_batch(
        self,
        texts: list[str],
        log_every: int = 100,
    ) -> list[dict]:
        """
        Classify a list of messages sequentially.

        Each message uses its own batched forward pass (all labels in one
        model call).  Cross-message batching is intentionally avoided because
        padding across different message lengths costs more than it saves at
        the dashboard's per-message throughput level.
        """
        results = []
        for i, text in enumerate(texts):
            results.append(self.classify(text))
            if log_every and (i + 1) % log_every == 0:
                logger.info("  classified %d / %d", i + 1, len(texts))
        return results

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _batched_forward(self, text: str) -> dict[str, float]:
        """
        Single forward pass: all label hypotheses batched against `text`.

        Repeats the premise N times, pairs with N hypothesis strings,
        tokenises as a batch, runs one forward pass, slices the entailment
        column (self._entailment_idx — set at load time from id2label).

        Returns {label: raw_entailment_logit}.
        """
        hypotheses = [
            self.hypothesis_templates.get(
                lbl, f"This message is an example of {lbl}."
            )
            for lbl in self.labels
        ]
        premises = [text] * len(self.labels)

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # logits shape: (n_labels, 3)
        logits     = self.model(**inputs).logits
        entailment = logits[:, self._entailment_idx].cpu().numpy()   # shape: (N,)

        return dict(zip(self.labels, entailment.tolist()))

    @staticmethod
    def _softmax(raw: dict[str, float]) -> dict[str, float]:
        """
        Numerically stable softmax over raw entailment logits.

        Preserves relative confidence gaps correctly: a logit difference
        of 2.0 maps to a much larger probability gap than a difference of 0.2,
        which simple L1 normalisation does not do.
        """
        keys   = list(raw.keys())
        logits = np.array([raw[k] for k in keys], dtype=np.float64)
        logits -= logits.max()           # shift for numerical stability
        exp    = np.exp(logits)
        probs  = exp / exp.sum()
        return dict(zip(keys, probs.tolist()))