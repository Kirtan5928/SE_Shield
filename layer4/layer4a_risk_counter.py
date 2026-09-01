"""
layer4/layer4a_risk_counter.py
================================
Layer 4a — Risk Counter (sits between Layer 2 and Layer 3).

Maintains a per-conversation sliding window of Layer 2 risk scores.
Outputs accumulated_risk and suspicious_flag to Layer 3, which uses them to:
  1. Override the SVM fast-path for suspicious conversations
     (catches trust-building messages that look benign in isolation)
  2. Lower the NLI confidence threshold when conversation is already flagged

Phase 1 (current): conversation_id supplied by caller — synthetic test convs.
Phase 2 (production): plug in real conversation_id sources below without
                      changing any internal logic.

PHASE 2 WIRING GUIDE
---------------------
Email    : conversation_id = email_thread_id  OR  hash(sender + recipient)
Chat     : conversation_id = session_id       OR  room_id
SMS      : conversation_id = hash(from_number + to_number)
API call : conversation_id = passed in request body as-is

Only the CALLER changes. RiskCounter internals are unchanged.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE          = 10     # max messages tracked per conversation
ATTACK_WEIGHT        = 1.0    # multiplier on (risk_score / 100) for attack msgs
BENIGN_DECAY         = 0.5    # subtracted from accumulator on benign msgs
FLOOR                = 0.0    # accumulator never goes below this
SUSPICIOUS_THRESHOLD = 3.0    # accumulated_risk above this → suspicious_flag=True

# Layer 3 threshold adjustment when conversation is suspicious
NORMAL_MIN_CONF     = 0.25    # default Layer 3 MIN_SUBTYPE_CONFIDENCE
SUSPICIOUS_MIN_CONF = 0.15    # lowered when suspicious_flag=True


# ---------------------------------------------------------------------------
# Per-conversation window state
# ---------------------------------------------------------------------------

@dataclass
class ConversationWindow:
    """
    State for one conversation.

    Phase 2: if you need persistence across server restarts, replace the
    deque with a Redis list or DB rows. The interface to RiskCounter
    (push / get_context) remains identical.
    """
    conversation_id: str
    window: Deque[float]      = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    accumulated_risk: float   = FLOOR
    message_count: int        = 0
    last_updated: float       = field(default_factory=time.time)

    def push(self, delta: float) -> None:
        """Add a risk delta (positive for attack, negative for benign)."""
        self.window.append(delta)
        self.accumulated_risk = max(FLOOR, self.accumulated_risk + delta)
        self.message_count   += 1
        self.last_updated     = time.time()

    @property
    def suspicious(self) -> bool:
        return self.accumulated_risk >= SUSPICIOUS_THRESHOLD

    @property
    def recommended_min_conf(self) -> float:
        return SUSPICIOUS_MIN_CONF if self.suspicious else NORMAL_MIN_CONF


# ---------------------------------------------------------------------------
# Risk Counter
# ---------------------------------------------------------------------------

class RiskCounter:
    """
    Layer 4a — per-conversation risk accumulator.

    Usage
    -----
    counter = RiskCounter()

    # For each incoming message (after Layer 2, before Layer 3):
    ctx = counter.update(
        conversation_id = "conv_42",
        svm_label       = "suspicious",   # or "benign"
        risk_score      = 78,             # Layer 2 LR output 0–100
    )

    # ctx dict is passed to Layer 3's run():
    result = layer3_pipeline.run(
        text              = clean_text,
        layer2_risk_score = risk_score,
        layer2_label      = svm_label if not ctx["override_svm"] else "suspicious",
        message_id        = message_id,
        timestamp         = timestamp,
        accumulated_risk  = ctx["accumulated_risk"],
        suspicious_flag   = ctx["suspicious_flag"],
    )

    Phase 2
    -------
    Replace __init__ with dependency-injected storage backend:

        def __init__(self, storage_backend=None):
            self._store = storage_backend or InMemoryStore()

    InMemoryStore (current) → RedisStore / SQLiteStore in production.
    The update() / get_context() / reset() interface never changes.
    """

    def __init__(
        self,
        window_size:          int   = WINDOW_SIZE,
        attack_weight:        float = ATTACK_WEIGHT,
        benign_decay:         float = BENIGN_DECAY,
        suspicious_threshold: float = SUSPICIOUS_THRESHOLD,
    ) -> None:
        self.window_size          = window_size
        self.attack_weight        = attack_weight
        self.benign_decay         = benign_decay
        self.suspicious_threshold = suspicious_threshold

        # In-memory store: conv_id → ConversationWindow
        # Phase 2: replace with RedisStore, SQLiteStore, etc.
        self._windows: dict[str, ConversationWindow] = {}

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def update(
        self,
        conversation_id: str,
        svm_label:       str,
        risk_score:      int,
    ) -> dict:
        """
        Process one message. Updates the conversation window and returns
        a context dict for Layer 3.

        Parameters
        ----------
        conversation_id : Unique conversation identifier.
                          Phase 1: synthetic e.g. "conv_42"
                          Phase 2: thread_id / session_id / hash(from+to)
        svm_label       : "benign" | "suspicious" from Layer 2 SVM.
        risk_score      : 0–100 from Layer 2 LR.

        Returns
        -------
        {
          "accumulated_risk"   : float  — running conversation risk score
          "suspicious_flag"    : bool   — True if accumulated_risk >= threshold
          "override_svm"       : bool   — True if SVM said benign but conv is
                                          suspicious → Layer 3 should run NLI
          "recommended_min_conf": float — adjusted Layer 3 confidence threshold
          "window_size"        : int    — number of messages seen so far
          "conversation_id"    : str    — echoed back for traceability
        }
        """
        win = self._get_or_create(conversation_id)

        # Compute delta
        if svm_label == "benign":
            delta = -self.benign_decay
        else:
            delta = (risk_score / 100.0) * self.attack_weight

        win.push(delta)

        # SVM override: conv is suspicious but this message looked benign
        override_svm = (svm_label == "benign") and win.suspicious

        logger.debug(
            "conv=%s  svm=%s  rs=%d  delta=%.2f  acc=%.2f  suspicious=%s  override=%s",
            conversation_id, svm_label, risk_score,
            delta, win.accumulated_risk, win.suspicious, override_svm,
        )

        return {
            "accumulated_risk":    round(win.accumulated_risk, 3),
            "suspicious_flag":     win.suspicious,
            "override_svm":        override_svm,
            "recommended_min_conf": win.recommended_min_conf,
            "window_size":         win.message_count,
            "conversation_id":     conversation_id,
        }

    def get_context(self, conversation_id: str) -> dict | None:
        """
        Read current context for a conversation without updating it.
        Returns None if conversation not yet seen.
        """
        win = self._windows.get(conversation_id)
        if not win:
            return None
        return {
            "accumulated_risk":    round(win.accumulated_risk, 3),
            "suspicious_flag":     win.suspicious,
            "recommended_min_conf": win.recommended_min_conf,
            "window_size":         win.message_count,
            "conversation_id":     conversation_id,
        }

    def reset(self, conversation_id: str) -> None:
        """
        Clear state for a conversation (e.g. after resolution or timeout).
        Phase 2: propagate delete to persistent storage backend.
        """
        self._windows.pop(conversation_id, None)
        logger.info("Reset conversation window: %s", conversation_id)

    def reset_all(self) -> None:
        """Clear all conversation state. Useful for test teardown."""
        self._windows.clear()

    def list_active(self) -> list[dict]:
        """Return summary of all active conversation windows."""
        return [
            {
                "conversation_id": cid,
                "accumulated_risk": round(w.accumulated_risk, 3),
                "suspicious":       w.suspicious,
                "message_count":    w.message_count,
            }
            for cid, w in self._windows.items()
        ]

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _get_or_create(self, conversation_id: str) -> ConversationWindow:
        if conversation_id not in self._windows:
            self._windows[conversation_id] = ConversationWindow(
                conversation_id=conversation_id,
                window=deque(maxlen=self.window_size),
            )
        return self._windows[conversation_id]