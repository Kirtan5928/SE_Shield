"""
menubar_app/accessibility.py
==============================
Reads the currently open conversation contact name from macOS Messages.app
via the Accessibility API, with AppleScript as a fallback.

Permission required: System Preferences → Privacy & Security → Accessibility
The menu bar app requests this automatically on first run.

Strategy (tried in order):
  1. AppleScript — most reliable for Messages.app across macOS versions
  2. AX API window title — fast, works when conversation name is in title
  3. AX API tree walk — deep scan of Messages UI tree
  4. Returns None — caller shows manual contact input dialog
"""

from __future__ import annotations

import hashlib
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Permission check ──────────────────────────────────────────────────────────

def check_permission() -> bool:
    """
    Returns True if this process has been granted Accessibility permission.
    If False, call request_permission() to open System Prefs to the right pane.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except ImportError:
        logger.error("pyobjc-framework-ApplicationServices not installed.")
        return False


def request_permission() -> None:
    """Open System Preferences to the Accessibility pane."""
    subprocess.run([
        "open",
        "x-apple.systempreferences:"
        "com.apple.preference.security?Privacy_Accessibility",
    ])


# ── Public interface ──────────────────────────────────────────────────────────

def get_messages_contact() -> Optional[str]:
    """
    Return the name of the currently open Messages conversation.

    Tries AppleScript first (most reliable), then falls back to the
    Accessibility API window title walk.

    Returns None if Messages is not open, no conversation is selected,
    or all strategies fail — caller should prompt for manual input.
    """
    # Strategy 1: AppleScript — reads directly from Messages app model
    contact = _applescript_contact()
    if contact:
        logger.debug("Contact from AppleScript: %s", contact)
        return contact

    # Strategy 2: AX API — window title / focused element
    contact = _ax_contact()
    if contact:
        logger.debug("Contact from AX API: %s", contact)
        return contact

    logger.debug("Could not auto-detect Messages contact.")
    return None


def get_whatsapp_desktop_contact() -> Optional[str]:
    """
    Return the currently open WhatsApp Desktop conversation name via AX API.
    WhatsApp Desktop is an Electron app — its UI is accessible via AX.
    """
    try:
        from AppKit import NSWorkspace
        pid = _find_pid("WhatsApp")
        if not pid:
            return None
        return _ax_window_title(pid, exclude={"WhatsApp"})
    except Exception as e:
        logger.debug("WA Desktop AX error: %s", e)
        return None


# ── Strategy 1: AppleScript ───────────────────────────────────────────────────

def _applescript_contact() -> Optional[str]:
    """
    Use AppleScript to get the most recently active Messages chat name.
    Works reliably across macOS Ventura, Sonoma, Sequoia.

    Requires Messages.app to be running (not necessarily in foreground).
    """
    script = '''
    tell application "Messages"
        if (count of chats) > 0 then
            set c to chat 1
            try
                return name of c
            on error
                -- Group chats have no single name; participants list instead
                set ps to participants of c
                set nameList to {}
                repeat with p in ps
                    set end of nameList to (name of p)
                end repeat
                return (nameList as string)
            end try
        end if
        return ""
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip()
        if result.returncode == 0 and name:
            return name
    except Exception as e:
        logger.debug("AppleScript error: %s", e)
    return None


# ── Strategy 2: Accessibility API ────────────────────────────────────────────

def _ax_contact() -> Optional[str]:
    """
    Walk the Messages app AX UI tree to find the conversation title.

    Tries:
    a) Window title (sometimes shows contact name)
    b) Focused element walk — find the conversation header text
    """
    try:
        pid = _find_pid_by_bundle("com.apple.MobileSMS")
        if not pid:
            return None

        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            kAXWindowsAttribute,
            kAXTitleAttribute,
            kAXFocusedUIElementAttribute,
        )

        ax_app = AXUIElementCreateApplication(pid)

        # a) Window title
        err, windows = AXUIElementCopyAttributeValue(
            ax_app, kAXWindowsAttribute, None
        )
        if err == 0 and windows:
            for win in (windows or []):
                err2, title = AXUIElementCopyAttributeValue(
                    win, kAXTitleAttribute, None
                )
                if err2 == 0 and title:
                    cleaned = title.strip()
                    # Filter out generic titles
                    if cleaned and cleaned.lower() not in ("messages", ""):
                        return cleaned

        # b) Walk children of first window looking for the conversation header
        if err == 0 and windows:
            contact = _ax_find_conversation_title(windows[0])
            if contact:
                return contact

    except Exception as e:
        logger.debug("AX API error: %s", e)
    return None


def _ax_find_conversation_title(element, depth: int = 0) -> Optional[str]:
    """
    Recursively search an AX element tree for a text node that looks like
    a contact name (not a generic UI label).
    Depth-limited to avoid excessive recursion on large UI trees.
    """
    if depth > 6:
        return None
    try:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            kAXRoleAttribute,
            kAXChildrenAttribute,
            kAXValueAttribute,
            kAXTitleAttribute,
            kAXDescriptionAttribute,
        )

        # Read role
        err, role = AXUIElementCopyAttributeValue(
            element, kAXRoleAttribute, None
        )
        role = role or ""

        # If this is a static text, check if it looks like a contact name
        if "StaticText" in role or "Heading" in role:
            for attr in (kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute):
                err, val = AXUIElementCopyAttributeValue(element, attr, None)
                if err == 0 and val:
                    s = str(val).strip()
                    if _looks_like_contact(s):
                        return s

        # Recurse into children
        err, children = AXUIElementCopyAttributeValue(
            element, kAXChildrenAttribute, None
        )
        if err == 0 and children:
            for child in (children or []):
                result = _ax_find_conversation_title(child, depth + 1)
                if result:
                    return result

    except Exception:
        pass
    return None


def _ax_window_title(pid: int, exclude: set = None) -> Optional[str]:
    """Read the main window title for a given PID."""
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            kAXWindowsAttribute,
            kAXTitleAttribute,
        )
        ax_app = AXUIElementCreateApplication(pid)
        err, windows = AXUIElementCopyAttributeValue(
            ax_app, kAXWindowsAttribute, None
        )
        if err == 0 and windows:
            for win in (windows or []):
                err2, title = AXUIElementCopyAttributeValue(
                    win, kAXTitleAttribute, None
                )
                if err2 == 0 and title:
                    t = title.strip()
                    if t and (not exclude or t not in exclude):
                        return t
    except Exception:
        pass
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_pid_by_bundle(bundle_id: str) -> Optional[int]:
    """Find running app PID by bundle identifier."""
    try:
        from AppKit import NSWorkspace
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if app.bundleIdentifier() == bundle_id:
                return int(app.processIdentifier())
    except Exception:
        pass
    return None


def _find_pid(app_name: str) -> Optional[int]:
    """Find running app PID by localised name."""
    try:
        from AppKit import NSWorkspace
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            name = app.localizedName() or ""
            if app_name.lower() in name.lower():
                return int(app.processIdentifier())
    except Exception:
        pass
    return None


_SKIP_LABELS = {
    "messages", "imessage", "ok", "cancel", "send", "new message",
    "compose", "back", "search", "done", "edit", "details",
    "", "whatsapp",
}

def _looks_like_contact(text: str) -> bool:
    """
    Heuristic: does this string look like a contact name rather than a UI label?
    Rejects generic button/section labels, very short strings, and known UI words.
    """
    if not text or len(text) < 2 or len(text) > 80:
        return False
    if text.lower() in _SKIP_LABELS:
        return False
    # Reject strings that are all digits (phone number raw format)
    if text.replace("+", "").replace("-", "").replace(" ", "").isdigit():
        return False
    return True