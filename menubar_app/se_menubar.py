"""
menubar_app/se_menubar.py
==========================
SE Shield — macOS Menu Bar App

Provides SE attack detection for macOS Messages (iMessage) and
WhatsApp Desktop by scanning clipboard text against the local
detection server at http://127.0.0.1:8000.

Usage
-----
  # From project root:
  python menubar_app/se_menubar.py

  # Or run as background process:
  nohup python menubar_app/se_menubar.py &

Workflow
--------
  1. Receive a suspicious message in Messages or WhatsApp Desktop
  2. Select the message text and copy it (Cmd+C)
  3. Click the 🛡 menu bar icon → "SCAN CLIPBOARD"
  4. A macOS notification appears with the alert level and reason
  5. Each scan from the same contact accumulates in the sliding window
     → multi-step SE attacks (trust_build_then_exploit, etc.) detected

Requirements
------------
  pip install -r menubar_app/requirements_menubar.txt --break-system-packages

Permissions needed (System Preferences → Privacy & Security):
  Accessibility  — reads contact name from Messages/WA Desktop UI
  Contacts       — resolves contact name to canonical email for cross-platform linking
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests
import pyperclip
import rumps

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # menubar_app/
_PROJECT_ROOT = _HERE.parent                      # hybrid_se/
sys.path.insert(0, str(_PROJECT_ROOT))

from menubar_app.accessibility import (
    check_permission,
    request_permission,
    get_messages_contact,
    get_whatsapp_desktop_contact,
)
from menubar_app.identity import resolve, check_contacts_permission, request_contacts_permission

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("se_menubar")

# ── Config ────────────────────────────────────────────────────────────────────
SERVER = "http://127.0.0.1:8000"

ALERT_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🔴",
    "MEDIUM":   "🟠",
    "LOW":      "🟢",
}

# ── Menu bar app ──────────────────────────────────────────────────────────────

class SEShieldApp(rumps.App):
    """
    SE Shield menu bar app.

    State
    -----
    _override_contact : str | None
      Manual contact name override — set by user via "Override Contact…" dialog.
      Takes priority over auto-detected name from Accessibility API.
      Cleared when user clicks "Clear Override".

    _last_conv_id : str | None
      The conversation_id used in the last scan. Stored so the user can
      reset the sliding window for a specific conversation.
    """

    def __init__(self):
        super().__init__("🛡", quit_button="Quit SE Shield")
        self._override_contact: Optional[str] = None
        self._last_conv_id:     Optional[str] = None
        self._check_permissions_on_start()

    # ── Permission check on startup ───────────────────────────────────────────

    def _check_permissions_on_start(self) -> None:
        """Request all required permissions on first run."""
        # Accessibility — needed to read contact name from Messages/WA Desktop
        if not check_permission():
            rumps.notification(
                "SE Shield — Permission Required",
                "Accessibility access needed",
                "Opening System Preferences → Accessibility. Add SE Shield.",
            )
            request_permission()

        # Contacts — needed for cross-platform identity resolution.
        # requestAccessForEntityType_ triggers the macOS system dialog.
        # Only fires if status is NotDetermined (0) — won't re-prompt if denied.
        if not check_contacts_permission():
            request_contacts_permission()

    # ── Menu items ────────────────────────────────────────────────────────────

    @rumps.clicked("Scan Clipboard")
    def scan_clipboard(self, _) -> None:
        """Main scan action — reads clipboard and sends to detection server."""

        # 1. Read clipboard
        try:
            text = pyperclip.paste().strip()
        except Exception:
            rumps.notification("SE Shield", "Clipboard error",
                               "Could not read clipboard.")
            return

        if len(text) < 10:
            rumps.notification("SE Shield", "Nothing to scan",
                               "Copy a message first (Cmd+C), then scan.")
            return

        # 2. Detect which app is frontmost + get contact name
        contact, platform = self._detect_contact_and_platform()

        # 3. Resolve contact to conversation_id
        conv_id = self._resolve_conv_id(contact, platform)
        self._last_conv_id = conv_id

        logger.info("Scanning — platform=%s contact=%s conv_id=%s",
                    platform, contact, conv_id)

        # 4. Send to server
        self._send_scan(text, conv_id, contact, platform)

    @rumps.clicked("Override Contact…")
    def override_contact(self, _) -> None:
        """
        Manually set the contact name for the current conversation.
        Use this when auto-detection fails or to set a specific contact
        before scanning a series of messages.
        """
        response = rumps.Window(
            message="Enter the contact name or phone number:",
            title="SE Shield — Set Contact",
            default_text=self._override_contact or "",
            ok="Set",
            cancel="Cancel",
        ).run()

        if response.clicked:
            name = response.text.strip()
            if name:
                self._override_contact = name
                rumps.notification(
                    "SE Shield", "Contact set",
                    f"Scanning as conversation with: {name}",
                )
            else:
                self._override_contact = None
                rumps.notification("SE Shield", "Contact cleared",
                                   "Auto-detection will be used.")

    @rumps.clicked("Clear Override")
    def clear_override(self, _) -> None:
        self._override_contact = None
        rumps.notification("SE Shield", "Override cleared",
                           "Auto-detection active.")

    @rumps.clicked("Reset Conversation Window")
    def reset_window(self, _) -> None:
        """Reset the sliding window for the last scanned conversation."""
        if not self._last_conv_id:
            rumps.notification("SE Shield", "Nothing to reset",
                               "Scan a message first.")
            return
        try:
            r = requests.delete(
                f"{SERVER}/conversation/{self._last_conv_id}", timeout=5
            )
            if r.status_code == 200:
                rumps.notification("SE Shield", "Window reset",
                                   f"Conversation {self._last_conv_id} cleared.")
                self._last_conv_id = None
            else:
                rumps.notification("SE Shield", "Reset failed",
                                   f"Server returned {r.status_code}.")
        except Exception as e:
            rumps.notification("SE Shield", "Reset failed", str(e)[:80])

    @rumps.clicked("Open Dashboard")
    def open_dashboard(self, _) -> None:
        """Open the Streamlit analyst dashboard in the browser."""
        import subprocess
        subprocess.run(["open", "http://127.0.0.1:8501"])

    @rumps.clicked("Check Server")
    def check_server(self, _) -> None:
        """Ping the local server to verify it is running."""
        try:
            r = requests.get(f"{SERVER}/health", timeout=5)
            if r.ok:
                d = r.json()
                rumps.notification(
                    "SE Shield — Server OK",
                    f"Active conversations: {d.get('active_conversations', 0)}",
                    f"Startup time: {d.get('startup_time_s', '?')}s",
                )
            else:
                raise ValueError(f"Status {r.status_code}")
        except Exception as e:
            rumps.notification(
                "SE Shield — Server Offline",
                "Run in terminal:",
                "uvicorn server.main:app --host 127.0.0.1 --port 8000",
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_contact_and_platform(self) -> tuple[Optional[str], str]:
        """
        Determine the contact name and platform for the current scan.

        Priority:
          1. User-set override contact (always wins)
          2. Auto-detect from Messages.app via Accessibility / AppleScript
          3. Auto-detect from WhatsApp Desktop via Accessibility
          4. None — caller will use hash-of-text fallback
        """
        if self._override_contact:
            # Guess platform from override source context
            return self._override_contact, "imessage"

        # Try Messages app first
        contact = get_messages_contact()
        if contact:
            return contact, "imessage"

        # Try WhatsApp Desktop
        contact = get_whatsapp_desktop_contact()
        if contact:
            return contact, "whatsapp_desktop"

        return None, "unknown"

    def _resolve_conv_id(
        self,
        contact: Optional[str],
        platform: str,
    ) -> str:
        """
        Resolve contact + platform to a stable conversation_id.

        If Contacts permission is granted: tries CNContactStore lookup
        → canonical email → entity_id (cross-platform stable).
        Otherwise: hashes the contact name directly (per-platform stable).
        """
        if contact:
            return resolve(name=contact)
        # No contact detected — use a text-independent platform fallback
        # (each scan isolated, window won't accumulate — user should set contact)
        import time
        return f"{platform}_{int(time.time() // 3600)}"  # groups scans within same hour

    def _send_scan(
        self,
        text:     str,
        conv_id:  str,
        contact:  Optional[str],
        platform: str,
    ) -> None:
        """POST scan request to local server and show notification."""
        payload = {
            "text":            text,
            "conversation_id": conv_id,
            "sender_name":     contact,
            "sender_email":    None,   # Phase 2: identity.py may resolve this
            "platform":        platform,
        }

        try:
            r = requests.post(f"{SERVER}/scan", json=payload, timeout=30)
            r.raise_for_status()
            self._notify_result(r.json(), contact)

        except requests.exceptions.ConnectionError:
            rumps.notification(
                "SE Shield — Server Offline",
                "Start the server first:",
                "uvicorn server.main:app --host 127.0.0.1 --port 8000",
            )
        except requests.exceptions.Timeout:
            rumps.notification("SE Shield", "Scan timed out",
                               "Server took too long. Is the NLI model loaded?")
        except Exception as e:
            rumps.notification("SE Shield", "Scan error", str(e)[:100])

    @staticmethod
    def _notify_result(data: dict, contact: Optional[str]) -> None:
        """Format and show macOS notification with scan result."""
        level   = data.get("alert_level",   "LOW")
        risk    = data.get("entity_risk",    0)
        pattern = data.get("attack_pattern", "none").replace("_", " ").upper()
        win_sz  = data.get("window_size",    0)
        last    = data.get("last_message",   {}) or {}
        reason  = (last.get("reason") or "").replace(" (low confidence:.*", "")[:100]
        label   = (last.get("label")  or "").replace("_", " ").upper()
        emoji   = ALERT_EMOJI.get(level, "🔵")

        contact_str = f" · {contact}" if contact else ""

        rumps.notification(
            title    = f"{emoji} SE Shield — {level}{contact_str}",
            subtitle = f"Risk {risk}/100 · {label} · Window {win_sz} msg{'s' if win_sz != 1 else ''}",
            message  = reason or pattern,
        )

        logger.info(
            "Result: level=%s risk=%d pattern=%s conv_window=%d",
            level, risk, pattern, win_sz,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEShieldApp().run()