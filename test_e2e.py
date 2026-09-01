#!/usr/bin/env python3
"""
test_e2e.py  (project root)
============================
End-to-end integration test for the full SE Shield stack.
Tests the FastAPI server with real HTTP requests.

Requires the server to be running:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

Run:
    python test_e2e.py
    python test_e2e.py --server http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys

import requests

G     = "\033[92m"
R     = "\033[91m"
B     = "\033[94m"
RESET = "\033[0m"
SEP   = "=" * 60

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  {G}+{RESET}  {name}")
    else:
        failed += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  {R}-{RESET}  {name}{suffix}")


def section(title: str) -> None:
    print()
    print(f"{B}── {title} ──{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SE Shield end-to-end integration test"
    )
    parser.add_argument(
        "--server", default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000)",
    )
    S = parser.parse_args().server

    print()
    print(SEP)
    print("  SE SHIELD — END-TO-END INTEGRATION TEST")
    print(SEP)

    # ── T1: Server health ─────────────────────────────────────────────────────
    section("Server Health")
    try:
        r = requests.get(f"{S}/health", timeout=5)
        check("Server reachable", r.status_code == 200)
        data = r.json()
        check("Pipeline ready", data.get("pipeline_ready", False),
              "Make sure uvicorn is running and pipeline loaded")
        check("Health returns startup_time_s", "startup_time_s" in data)
    except requests.exceptions.ConnectionError:
        print(f"  {R}FATAL{RESET}  Cannot reach {S}")
        print("  Start the server first:")
        print("  uvicorn server.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # ── T2: Single message scan — clear phishing ──────────────────────────────
    section("Single Message Scan")
    phishing_text = (
        "URGENT: Your account will be suspended. "
        "Click here to verify your credentials immediately."
    )
    r = requests.post(
        f"{S}/scan",
        json={
            "text":            phishing_text,
            "conversation_id": "e2e_test_phishing",
            "message_id":      "m1",
        },
        timeout=30,
    )
    check("Scan returns 200", r.status_code == 200, str(r.status_code))

    data = r.json()
    check("Returns entity_risk",    "entity_risk"    in data)
    check("Returns alert_level",    "alert_level"    in data)
    check("Returns attack_pattern", "attack_pattern" in data)
    check("Returns reasons list",   isinstance(data.get("reasons"), list))
    check("Returns last_message",   data.get("last_message") is not None)

    last_label = data.get("last_message", {}).get("label", "")
    check("Attack detected — label is not benign",
          last_label != "benign",
          f"got label={last_label}")

    alert = data.get("alert_level", "")
    check("Alert is MEDIUM / HIGH / CRITICAL for clear phishing",
          alert in ("MEDIUM", "HIGH", "CRITICAL"),
          f"got alert_level={alert}")

    # ── T3: Benign message ────────────────────────────────────────────────────
    section("Benign Message")
    benign_text = (
        "Hi, just wanted to follow up on the meeting. "
        "Please review the attached document when you get a chance."
    )
    r2 = requests.post(
        f"{S}/scan",
        json={
            "text":            benign_text,
            "conversation_id": "e2e_test_benign",
        },
        timeout=30,
    )
    check("Benign scan returns 200", r2.status_code == 200)

    d2        = r2.json()
    lbl_benign = d2.get("last_message", {}).get("label", "")
    check("Benign message labelled benign",
          lbl_benign == "benign",
          f"got label={lbl_benign}")

    # ── T4: Sliding window accumulates across messages ────────────────────────
    section("Sliding Window Accumulation")
    conv = "e2e_test_conv_001"
    msgs = [
        "Hi, I am from IT helpdesk. We detected unusual activity on your account.",
        "This is just a routine security check, nothing to worry about.",
        "We need your login credentials to complete the verification process.",
    ]
    results = []
    for i, text in enumerate(msgs):
        r3 = requests.post(
            f"{S}/scan",
            json={
                "text":            text,
                "conversation_id": conv,
                "message_id":      f"m{i + 1}",
            },
            timeout=30,
        )
        check(f"Message {i + 1} scans without error", r3.status_code == 200,
              r3.text[:80] if r3.status_code != 200 else "")
        results.append(r3.json())

    final_win = results[-1].get("window_size", 0)
    check("Window size >= 2 after 3 messages",
          final_win >= 2,
          f"got window_size={final_win}")

    final_alert = results[-1].get("alert_level", "")
    check("Final alert is HIGH or CRITICAL",
          final_alert in ("HIGH", "CRITICAL"),
          f"got alert_level={final_alert}")

    final_pattern = results[-1].get("attack_pattern", "")
    check("Attack pattern detected",
          final_pattern not in ("none", "no_attack_detected"),
          f"got attack_pattern={final_pattern}")

    reasons_len = len(results[-1].get("reasons", []))
    check("Reasons trail has entries",
          reasons_len > 0,
          f"got {reasons_len} reasons")

    # ── T5: GET conversation (read-only snapshot) ─────────────────────────────
    section("Conversation Read")
    r4 = requests.get(f"{S}/conversation/{conv}", timeout=10)
    check("GET /conversation returns 200", r4.status_code == 200,
          str(r4.status_code))

    snap_win = r4.json().get("window_size", -1)  # server returns flat ConversationAssessment
    check("GET window_size matches scan window_size",
          snap_win == final_win,
          f"expected {final_win} got {snap_win}")

    # ── T6: DELETE conversation (reset) ──────────────────────────────────────
    section("Conversation Reset")
    r5 = requests.delete(f"{S}/conversation/{conv}", timeout=10)
    check("DELETE returns 200", r5.status_code == 200, str(r5.status_code))

    r6 = requests.get(f"{S}/conversation/{conv}", timeout=10)
    check("Window gone after reset — GET returns 404",
          r6.status_code == 404,
          f"got status={r6.status_code}")

    # ── T7: Conversations list ────────────────────────────────────────────────
    section("Conversations List")
    r7 = requests.get(f"{S}/conversations", timeout=10)
    check("GET /conversations returns 200", r7.status_code == 200)
    check("Response contains 'count' field", "count" in r7.json())

    # ── T8: sender_name/email overrides raw conv_id via identity resolution ──
    section("Identity Resolution")
    r8 = requests.post(f"{S}/scan", json={
        "text":            "Urgent: your account will be suspended",
        "conversation_id": "raw_platform_id_should_be_overridden",
        "sender_email":    "identity.test@example.com",
        "platform":        "whatsapp_web",
    }, timeout=30)
    check("sender_email overrides raw conv_id",
          r8.status_code == 200 and
          r8.json().get("conversation_id", "").startswith("entity_"),
          f"got conv_id={r8.json().get('conversation_id')}")

    # ── T9: Same sender_email on two platforms → same entity_id → shared window
    # Uses sender_email on both requests — no Contacts lookup required,
    # email is canonical so match is guaranteed in any environment.
    section("Cross-Platform Shared Window")
    shared_email = "crossplatform.se.test@example.com"

    # Clean up any leftover state
    from menubar_app.identity import resolve as _id_resolve
    entity_id = _id_resolve(email=shared_email)
    requests.delete(f"{S}/conversation/{entity_id}", timeout=5)

    r9a = requests.post(f"{S}/scan", json={
        "text":            "Hi I am from IT helpdesk, we detected unusual activity",
        "conversation_id": "imsg_cp_test_raw",
        "sender_email":    shared_email,
        "platform":        "imessage",
    }, timeout=30)
    r9b = requests.post(f"{S}/scan", json={
        "text":            "We need your login credentials to complete the verification",
        "conversation_id": "gmail_cp_test_raw",
        "sender_email":    shared_email,
        "platform":        "gmail",
    }, timeout=30)

    d9a, d9b = r9a.json(), r9b.json()
    check("Cross-platform: both scans return 200",
          r9a.status_code == 200 and r9b.status_code == 200,
          f"{r9a.status_code} / {r9b.status_code}")
    check("Cross-platform: same entity_id on both platforms",
          d9a.get("conversation_id") == d9b.get("conversation_id"),
          f"imsg={d9a.get('conversation_id')} gmail={d9b.get('conversation_id')}")
    check("Cross-platform: window_size=2 after 2 platforms",
          d9b.get("window_size") == 2,
          f"got {d9b.get('window_size')}  (expected 2)")
    check("Cross-platform: alert escalated after 2 messages",
          d9b.get("alert_level") in ("MEDIUM", "HIGH", "CRITICAL"),
          f"got {d9b.get('alert_level')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    colour = G if failed == 0 else R
    print(f"  {colour}{passed} passed  {failed} failed{RESET}")

    if results:
        final = results[-1]
        print()
        print("  Final conversation assessment:")
        print(f"    entity_risk    : {final.get('entity_risk')}/100")
        print(f"    alert_level    : {final.get('alert_level')}")
        print(f"    attack_pattern : {final.get('attack_pattern')}")
        print(f"    window_size    : {final.get('window_size')}")

    print(SEP)
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()