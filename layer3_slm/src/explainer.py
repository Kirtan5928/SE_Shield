"""
layer3_slm/src/explainer.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from config_layer3 import (
    URGENCY_SIGNALS, CREDENTIAL_SIGNALS, AUTHORITY_SIGNALS,
    FINANCIAL_SIGNALS, THREAT_SIGNALS, LINK_ACTION_SIGNALS
)

LOW_CONF_THRESHOLD = 0.35


@dataclass
class DetectedSignals:
    urgency:     list[str] = field(default_factory=list)
    credential:  list[str] = field(default_factory=list)
    authority:   list[str] = field(default_factory=list)
    financial:   list[str] = field(default_factory=list)
    threat:      list[str] = field(default_factory=list)
    link_action: list[str] = field(default_factory=list)


class SESignalExtractor:
    def __init__(
        self,
        urgency:     Sequence[str] | None = None,
        credential:  Sequence[str] | None = None,
        authority:   Sequence[str] | None = None,
        financial:   Sequence[str] | None = None,
        threat:      Sequence[str] | None = None,
        link_action: Sequence[str] | None = None,
    ) -> None:
        self._urgency     = list(urgency     or URGENCY_SIGNALS)
        self._credential  = list(credential  or CREDENTIAL_SIGNALS)
        self._authority   = list(authority   or AUTHORITY_SIGNALS)
        self._financial   = list(financial   or FINANCIAL_SIGNALS)
        self._threat      = list(threat      or THREAT_SIGNALS)
        self._link_action = list(link_action or LINK_ACTION_SIGNALS)

    def extract(self, text: str) -> DetectedSignals:
        t = text.lower()
        return DetectedSignals(
            urgency     = self._scan(self._urgency,     t),
            credential  = self._scan(self._credential,  t),
            authority   = self._scan(self._authority,   t),
            financial   = self._scan(self._financial,   t),
            threat      = self._scan(self._threat,      t),
            link_action = self._scan(self._link_action, t),
        )

    @staticmethod
    def _scan(patterns: list[str], text: str) -> list[str]:
        seen: set[str] = set()
        found: list[str] = []
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                snippet = m.group(0).strip()
                if snippet not in seen:
                    seen.add(snippet)
                    found.append(snippet)
        return found


class ExplanationEngine:
    _FRAMES: dict[str, str] = {
        "benign": "No social engineering signals detected — message appears legitimate.",

        "phishing": "Message {signals_clause} — pattern consistent with a phishing attempt.",

        "spear_phishing": "Message {signals_clause} and uses personal context — consistent with spear phishing.",

        "pretexting": "Message {signals_clause} using a fabricated scenario — consistent with pretexting.",

        "credential_harvesting": "Message {signals_clause} directing to a login form — consistent with credential harvesting.",

        "baiting": "Message {signals_clause} offering a free prize or gift — consistent with a baiting attack.",

        "vishing": "Message {signals_clause} instructing to call a number — consistent with vishing.",

        "business_email_compromise": "Message {signals_clause} requesting urgent financial transaction — consistent with Business Email Compromise (BEC).",
    }

    def __init__(self, extractor: SESignalExtractor | None = None) -> None:
        self.extractor = extractor or SESignalExtractor()

    def explain(self, text: str, clf: dict) -> str:
        label = clf["label"]
        confidence = clf["confidence"]

        if label == "benign":
            return self._FRAMES["benign"]

        signals = self.extractor.extract(text)
        signals_clause = self._build_clause(signals)

        frame = self._FRAMES.get(label, "Message {signals_clause} — flagged as suspicious.")
        reason = frame.format(signals_clause=signals_clause)

        if confidence < LOW_CONF_THRESHOLD:
            reason += f" (low confidence: {confidence:.2f} — manual review recommended)"

        return reason

    @staticmethod
    def _build_clause(signals: DetectedSignals) -> str:
        parts: list[str] = []

        if signals.urgency:
            parts.append(f"creates urgency ({', '.join(signals.urgency[:2])})")
        if signals.credential:
            parts.append(f"requests credentials ({', '.join(signals.credential[:2])})")
        if signals.authority:
            parts.append(f"impersonates authority ({', '.join(signals.authority[:1])})")
        if signals.financial:
            parts.append(f"requests financial action ({', '.join(signals.financial[:2])})")
        if signals.threat:
            parts.append(f"uses threats ({', '.join(signals.threat[:2])})")
        if signals.link_action:
            parts.append("directs to click/download/verify")

        if not parts:
            return "shows suspicious semantic intent according to NLI"

        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + ", and " + parts[-1]