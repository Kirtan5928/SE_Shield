"""
layer4/layer4b_semantic_window.py
===================================
Layer 4b — Semantic Pattern Detector (sits after Layer 3).

Maintains a per-conversation sliding window of Layer 3 output dicts.
Detects multi-step SE attack patterns that are invisible at the
per-message level.

Patterns detected
-----------------
1. urgency_escalation       — NLI confidence rising across window
2. authority_then_credential — pretexting followed by credential_harvesting
3. trust_build_then_exploit — benign × N then high-confidence attack
4. bec_sequence             — pretexting/spear_phishing then BEC
5. multi_vector             — 3+ different attack types in one window
6. delayed_execution        — long time gap then sudden high-confidence attack

Output contract (to Dashboard / Layer 5)
-----------------------------------------
{
  "conversation_id" : str
  "entity_risk"     : int   0–100
  "attack_pattern"  : str   matched pattern name or "mixed_attack"
  "alert_level"     : str   "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  "dominant_label"  : str   most frequent attack label in window
  "confidence"      : float weighted average L3 confidence
  "reasons"         : list[str]   per-message reasoning trail
  "window_size"     : int   messages in current window
  "message_ids"     : list[str | None]
}

Phase 2 wiring
--------------
Same as Layer 4a — only the storage backend changes. The detect() interface
and output contract are stable and can be consumed by any dashboard.

ENTITY RISK FORMULA (Problem 2 fix)
------------------------------------
The old base was a flat layer2_risk-weighted average of attack-message
confidences. In escalating conversations (Arjun script: small talk →
pressure → explicit credential request), later low-confidence attack
messages dragged the average DOWN — risk fell from 57 to 42 while the
conversation got objectively more dangerous, and never reached HIGH.

New base = blend of:
  * RECENCY-WEIGHTED average — each later attack message weighs
    RECENCY_GROWTH× more than the previous one, so the most recent
    (typically most explicit) message dominates the score.
  * PEAK confidence — once a high-confidence attack message has been
    seen, subsequent low-confidence noise cannot erase it. Risk is
    monotone-friendly: it can plateau, but it does not collapse.

base = ((1 - PEAK_BLEND) * recency_avg + PEAK_BLEND * peak) * 100
Pattern bonus and count bonus are unchanged.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE  = 10    # max L3 outputs tracked per conversation

# Entity risk thresholds → alert levels
CRITICAL_THRESHOLD = 80
HIGH_THRESHOLD     = 60
MEDIUM_THRESHOLD   = 35

# Pattern detection config
TRUST_BUILD_MIN_BENIGN    = 2     # min consecutive benign msgs before exploit
TRUST_BUILD_MIN_ATK_CONF  = 0.50  # exploit message min confidence
ESCALATION_MIN_RISE       = 0.20  # min confidence rise across window to flag
DELAYED_EXEC_MIN_MINS     = 30    # minutes gap that triggers delayed_execution flag
BEC_SEQUENCE_LABELS       = {"pretexting", "spear_phishing", "business_email_compromise"}
MULTI_VECTOR_MIN_TYPES    = 3     # distinct attack types to flag multi_vector

# Entity risk formula config (Problem 2 fix)
RECENCY_GROWTH = 1.25   # each later attack msg weighs 25% more than previous
PEAK_BLEND     = 0.45   # fraction of base taken from peak confidence
                        # 0.0 = pure recency average, 1.0 = pure peak hold


# ---------------------------------------------------------------------------
# Per-message slot (what gets stored in the window)
# ---------------------------------------------------------------------------

@dataclass
class MessageSlot:
    message_id:  str | None
    timestamp:   str | None
    label:       str
    confidence:  float
    top_labels:  list[dict]
    reason:      str
    layer2_risk: int


# ---------------------------------------------------------------------------
# Semantic Window
# ---------------------------------------------------------------------------

class SemanticWindow:
    """
    Layer 4b — per-conversation semantic sliding window.

    Usage
    -----
    window = SemanticWindow()

    # Feed each Layer 3 output:
    result = window.update("conv_42", layer3_output_dict)

    # result is the Layer 5 / dashboard input contract.
    # result["alert_level"] tells the dashboard what to display.

    Phase 2
    -------
    Replace in-memory deque with persistent storage (same as Layer 4a).
    detect() and update() interfaces are unchanged.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size
        # conv_id → deque of MessageSlot
        self._windows: dict[str, Deque[MessageSlot]] = {}

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def update(self, conversation_id: str, l3_output: dict) -> dict:
        """
        Add one Layer 3 output to the window and return the current
        conversation-level assessment.

        Parameters
        ----------
        conversation_id : Same ID used in Layer 4a.
        l3_output       : Full Layer 3 output contract dict.

        Returns
        -------
        Dashboard-ready output contract dict.
        """
        win = self._get_or_create(conversation_id)

        slot = MessageSlot(
            message_id  = l3_output.get("message_id"),
            timestamp   = l3_output.get("timestamp"),
            label       = l3_output.get("label", "phishing"),
            confidence  = l3_output.get("confidence", 0.0),
            top_labels  = l3_output.get("top_labels", []),
            reason      = l3_output.get("reason", ""),
            layer2_risk = l3_output.get("layer2_risk", 0),
        )
        win.append(slot)

        return self.assess(conversation_id)

    def assess(self, conversation_id: str) -> dict:
        """
        Compute current conversation-level risk assessment from window.
        Can be called without updating (read-only snapshot).
        """
        win = self._windows.get(conversation_id)
        if not win:
            return self._empty_assessment(conversation_id)

        slots = list(win)
        attack_slots = [s for s in slots if s.label != "benign"]

        pattern      = self._detect_pattern(slots)
        entity_risk  = self._compute_entity_risk(slots, pattern)
        alert_level  = self._alert_level(entity_risk)
        dominant     = self._dominant_label(attack_slots)
        avg_conf     = self._weighted_confidence(attack_slots)
        reasons      = self._build_reasons(slots)

        return {
            "conversation_id": conversation_id,
            "entity_risk":     entity_risk,
            "attack_pattern":  pattern,
            "alert_level":     alert_level,
            "dominant_label":  dominant,
            "confidence":      round(avg_conf, 4),
            "reasons":         reasons,
            "window_size":     len(slots),
            "message_ids":     [s.message_id for s in slots],
        }

    def reset(self, conversation_id: str) -> None:
        """Clear window for a conversation."""
        self._windows.pop(conversation_id, None)

    def reset_all(self) -> None:
        self._windows.clear()

    def list_active(self) -> list[dict]:
        return [
            {
                "conversation_id": cid,
                "window_size":     len(w),
                "alert_level":     self._alert_level(
                    self._compute_entity_risk(list(w), self._detect_pattern(list(w)))
                ),
            }
            for cid, w in self._windows.items()
        ]

    # ------------------------------------------------------------------ #
    # Pattern detectors                                                     #
    # ------------------------------------------------------------------ #

    def _detect_pattern(self, slots: list[MessageSlot]) -> str:
        """
        Check patterns in priority order. Return first match.
        Priority: specific patterns > generic fallbacks.
        """
        attack_slots = [s for s in slots if s.label != "benign"]
        if not attack_slots:
            return "no_attack_detected"

        # 1. trust_build_then_exploit
        if self._is_trust_build_then_exploit(slots):
            return "trust_build_then_exploit"

        # 2. authority_then_credential
        if self._is_authority_then_credential(slots):
            return "authority_then_credential"

        # 3. bec_sequence
        if self._is_bec_sequence(slots):
            return "bec_sequence"

        # 4. urgency_escalation
        if self._is_urgency_escalation(slots):
            return "urgency_escalation"

        # 5. delayed_execution
        if self._is_delayed_execution(slots):
            return "delayed_execution"

        # 6. multi_vector
        if self._is_multi_vector(attack_slots):
            return "multi_vector"

        # 7. fallback
        if len(attack_slots) == 1:
            return attack_slots[0].label
        return "mixed_attack"

    def _is_trust_build_then_exploit(self, slots: list[MessageSlot]) -> bool:
        """
        Detects: [benign, benign, ..., high-conf attack]
        Classic long-game SE sequence.
        """
        if len(slots) < TRUST_BUILD_MIN_BENIGN + 1:
            return False
        last = slots[-1]
        if last.label == "benign" or last.confidence < TRUST_BUILD_MIN_ATK_CONF:
            return False
        # Check if there were enough benign messages before the final attack
        prior_benign = sum(1 for s in slots[:-1] if s.label == "benign")
        return prior_benign >= TRUST_BUILD_MIN_BENIGN

    def _is_authority_then_credential(self, slots: list[MessageSlot]) -> bool:
        """
        Detects: pretexting/vishing appears before credential_harvesting.
        Classic IT-support impersonation → password steal sequence.
        """
        labels = [s.label for s in slots]
        authority_labels = {"pretexting", "vishing", "spear_phishing"}
        has_authority    = any(l in authority_labels for l in labels)
        has_cred         = "credential_harvesting" in labels
        if not (has_authority and has_cred):
            return False
        # Authority must appear BEFORE credential_harvesting
        first_authority = next(i for i, l in enumerate(labels) if l in authority_labels)
        first_cred      = next(i for i, l in enumerate(labels) if l == "credential_harvesting")
        return first_authority < first_cred

    def _is_bec_sequence(self, slots: list[MessageSlot]) -> bool:
        """
        Detects: labels from BEC_SEQUENCE_LABELS with BEC as the latest attack.
        """
        labels = [s.label for s in slots if s.label != "benign"]
        if not labels:
            return False
        bec_labels_present = set(labels) & BEC_SEQUENCE_LABELS
        return (
            len(bec_labels_present) >= 2
            and labels[-1] == "business_email_compromise"
        )

    def _is_urgency_escalation(self, slots: list[MessageSlot]) -> bool:
        """
        Detects: NLI confidence steadily rising across attack messages.
        """
        attack_confs = [s.confidence for s in slots if s.label != "benign"]
        if len(attack_confs) < 3:
            return False
        total_rise = attack_confs[-1] - attack_confs[0]
        return total_rise >= ESCALATION_MIN_RISE

    def _is_delayed_execution(self, slots: list[MessageSlot]) -> bool:
        """
        Detects: a significant time gap before the final high-confidence attack.
        Requires ISO-8601 timestamps on messages.
        """
        if len(slots) < 2:
            return False
        # Need at least first and last timestamps
        ts_slots = [s for s in slots if s.timestamp]
        if len(ts_slots) < 2:
            return False
        try:
            t_first = datetime.fromisoformat(ts_slots[0].timestamp.replace("Z", "+00:00"))
            t_last  = datetime.fromisoformat(ts_slots[-1].timestamp.replace("Z", "+00:00"))
            gap_mins = (t_last - t_first).total_seconds() / 60
            last_is_attack = slots[-1].label != "benign"
            last_is_confident = slots[-1].confidence >= TRUST_BUILD_MIN_ATK_CONF
            return gap_mins >= DELAYED_EXEC_MIN_MINS and last_is_attack and last_is_confident
        except (ValueError, AttributeError):
            return False

    def _is_multi_vector(self, attack_slots: list[MessageSlot]) -> bool:
        """Detects: 3+ distinct attack types in one conversation window."""
        distinct = {s.label for s in attack_slots}
        return len(distinct) >= MULTI_VECTOR_MIN_TYPES

    # ------------------------------------------------------------------ #
    # Risk scoring                                                          #
    # ------------------------------------------------------------------ #

    def _compute_entity_risk(
        self, slots: list[MessageSlot], pattern: str
    ) -> int:
        """
        Compute entity-level risk score 0–100.

        Formula (Problem 2 fix):
          base  = blend of recency-weighted confidence average and peak
                  confidence of attack messages × 100
          bonus = pattern severity bonus
          count = logarithmic bonus for number of attack messages

        Recency weighting: slot i weight = max(layer2_risk, 1) × RECENCY_GROWTH^i
          → the latest attack message dominates; an escalating conversation
            escalates instead of averaging out.
        Peak blending: PEAK_BLEND fraction of base comes from the highest
          confidence ever seen in the window → later low-confidence messages
          can plateau the score but never collapse it.
        """
        attack_slots = [s for s in slots if s.label != "benign"]
        if not attack_slots:
            return 0

        def _eff_conf(s: MessageSlot) -> float:
            # Blend NLI confidence with L2 calibrated probability.
            # When two independent models agree (NLI high + LR high),
            # effective confidence rises. When they disagree (NLI low
            # but LR high — as in split-softmax cases like Rajesh where
            # mass spreads across pretexting + credential_harvesting),
            # L2 provides a floor so the score reflects true threat level.
            l2 = max(s.layer2_risk, 0) / 100.0
            return max(s.confidence, 0.5 * (s.confidence + l2))

        # Base: recency-weighted effective-confidence average blended with peak
        weights = [
            max(s.layer2_risk, 1) * (RECENCY_GROWTH ** i)
            for i, s in enumerate(attack_slots)
        ]
        recency_avg = (
            sum(w * _eff_conf(s) for w, s in zip(weights, attack_slots))
            / sum(weights)
        )
        peak = max(_eff_conf(s) for s in attack_slots)
        base = ((1.0 - PEAK_BLEND) * recency_avg + PEAK_BLEND * peak) * 100

        # Pattern severity bonus
        pattern_bonus = {
            "trust_build_then_exploit":    20,
            "authority_then_credential":   18,
            "bec_sequence":                15,
            "urgency_escalation":          10,
            "delayed_execution":           12,
            "multi_vector":                15,
            "mixed_attack":                 5,
        }.get(pattern, 0)

        # Count bonus: more attack messages = higher risk
        count_bonus = min(10, math.log(len(attack_slots) + 1, 2) * 4)

        return min(100, int(base + pattern_bonus + count_bonus))

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _alert_level(entity_risk: int) -> str:
        if entity_risk >= CRITICAL_THRESHOLD:
            return "CRITICAL"
        if entity_risk >= HIGH_THRESHOLD:
            return "HIGH"
        if entity_risk >= MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _dominant_label(attack_slots: list[MessageSlot]) -> str:
        if not attack_slots:
            return "none"
        counts = Counter(s.label for s in attack_slots)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _weighted_confidence(attack_slots: list[MessageSlot]) -> float:
        if not attack_slots:
            return 0.0
        total = sum(s.layer2_risk for s in attack_slots)
        if total == 0:
            return sum(s.confidence for s in attack_slots) / len(attack_slots)
        return sum(s.confidence * s.layer2_risk for s in attack_slots) / total

    @staticmethod
    def _build_reasons(slots: list[MessageSlot]) -> list[str]:
        """Build human-readable reasoning trail for the dashboard."""
        reasons = []
        for i, s in enumerate(slots):
            mid   = s.message_id or f"msg_{i+1:03d}"
            ts    = f" [{s.timestamp}]" if s.timestamp else ""
            label = s.label
            conf  = f"{s.confidence:.2f}"
            note  = "— trust building" if label == "benign" else f"— {s.reason[:80]}"
            reasons.append(f"{mid}{ts}: {label} ({conf}) {note}")
        return reasons

    @staticmethod
    def _empty_assessment(conversation_id: str) -> dict:
        return {
            "conversation_id": conversation_id,
            "entity_risk":     0,
            "attack_pattern":  "no_attack_detected",
            "alert_level":     "LOW",
            "dominant_label":  "none",
            "confidence":      0.0,
            "reasons":         [],
            "window_size":     0,
            "message_ids":     [],
        }

    def _get_or_create(self, conversation_id: str) -> Deque[MessageSlot]:
        if conversation_id not in self._windows:
            self._windows[conversation_id] = deque(maxlen=self.window_size)
        return self._windows[conversation_id]