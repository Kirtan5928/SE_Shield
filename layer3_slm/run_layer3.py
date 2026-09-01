"""
run_layer3.py  (project root)
===============================
CLI demo and integration smoke test for Layer 3.

Usage
-----
  # Single message:
  python run_layer3.py -m "Urgent: your account will be suspended. Verify now."

  # From stdin:
  echo "Please wire the funds to the new account before EOD." | python run_layer3.py --stdin

  # With explicit Layer 2 risk score:
  python run_layer3.py -m "Download the attached invoice" --risk-score 72

  # Pipe in a Layer 2 JSON output dict:
  python run_layer3.py --layer2-output '{"text": "Confirm your password", "risk_score": 85}'

  # JSON output (for programmatic use):
  python run_layer3.py -m "Hi, let's catch up tomorrow" --json

  # Smoke test on 10 built-in examples:
  python run_layer3.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
_LAYER3_DIR = _ROOT / "layer3_slm"
for p in [str(_LAYER3_DIR), str(_ROOT)]:
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Smoke-test examples ───────────────────────────────────────────────────────
# Tuple: (description, text, expected_label, simulated_layer2_risk_score)
#
# Architecture note on benign risk scores:
#   Benign messages should have risk_score <= LAYER2_THRESHOLD (50) so they
#   are gated OUT before NLI — exactly as in the real pipeline.
#   The NLI model was pre-trained on MNLI data and is not designed to be the
#   benign/attack gate; that role belongs to Layer 2.  Forcing benign messages
#   through NLI at risk_score=100 is a category error in the smoke test.
SMOKE_EXAMPLES = [
    # (description, text, expected_label, layer2_risk_score)
    ("Phishing — urgency + account threat",
     "URGENT: Your account has been compromised. Click here to verify your credentials immediately or it will be suspended.",
     "phishing", 85),
    ("BEC — wire transfer request",
     "Hi Sarah, this is Mike from finance. We need to process an urgent wire transfer of $47,000 to our new vendor before EOD today. Can you action this?",
     "business_email_compromise", 80),
    ("Credential harvesting — fake login",
     "Your Netflix account requires verification. Please sign in at netflix-secure-verify.com to confirm your username and password.",
     "credential_harvesting", 82),
    ("Spear phishing — personalised",
     "Hi John, following our meeting last Tuesday about the Q3 budget, the CFO has asked me to forward this invoice for your approval. Please review and authorize.",
     "spear_phishing", 78),
    ("Pretexting — IT support impersonation",
     "This is the IT helpdesk. We have detected unusual activity on your machine and need your login credentials to run a remote diagnostic.",
     "pretexting", 84),
    ("Vishing — bank impersonation",
     "This is an automated security message from your bank. Suspicious activity has been detected. Please call our fraud line immediately and have your account number ready.",
     "vishing", 76),
    ("Baiting — free prize lure",
     "Congratulations! You have been selected to receive a free iPhone 15. Click the download link to claim your prize before it expires.",
     "baiting", 73),
    ("Benign — casual email [Layer 2 gated]",
     "Hey, are you free for lunch tomorrow? I was thinking we could try that new Italian place on Main Street.",
     "benign", 8),   # ← below threshold: NLI never runs
    ("Benign — work scheduling [Layer 2 gated]",
     "The 2pm meeting has been moved to 3pm. Please update your calendar. Conference room B is booked.",
     "benign", 5),   # ← below threshold: NLI never runs
    ("Subtle BEC — no urgency keywords",
     "Can you process the payment for the attached purchase order? The vendor has updated their banking details.",
     "business_email_compromise", 70),
]


def build_pipeline() -> Layer3Pipeline:
    return Layer3Pipeline(
        model_name=MODEL_NAME,
        labels=ATTACK_LABELS,
        hypothesis_templates=HYPOTHESIS_TEMPLATES,
        layer2_threshold=LAYER2_THRESHOLD,
        max_length=MAX_LENGTH,
    )


def pretty_print(result: dict, text: str | None = None) -> None:
    label = result["label"]
    conf  = result["confidence"]

    label_colour = {
        "benign":                    "\033[92m",
        "phishing":                  "\033[91m",
        "spear_phishing":            "\033[91m",
        "credential_harvesting":     "\033[91m",
        "business_email_compromise": "\033[91m",
        "pretexting":                "\033[93m",
        "baiting":                   "\033[93m",
        "vishing":                   "\033[93m",
    }.get(label, "\033[0m")
    reset = "\033[0m"

    print("\n" + "─" * 70)
    if text:
        print(f"  TEXT        : {text[:80]}{'…' if len(text) > 80 else ''}")

    # message_id and timestamp — shown only when present
    if result.get("message_id"):
        print(f"  MESSAGE ID  : {result['message_id']}")
    if result.get("timestamp"):
        print(f"  TIMESTAMP   : {result['timestamp']}")

    print(f"  LABEL       : {label_colour}{label.upper()}{reset}")
    print(f"  CONFIDENCE  : {conf:.4f}")

    top_labels = result.get("top_labels", [])
    if top_labels:
        print(f"\n  TOP LABELS  :")
        for i, t in enumerate(top_labels):
            lc = label_colour if t["label"] == label else ""
            print(f"    [{i}] {lc}{t['label']:<35}{reset}  {t['confidence']:.4f}")

    print(f"\n  REASON      : {result['reason']}")
    print(f"  L2 RISK     : {result['layer2_risk']}")
    print(f"  LATENCY     : {result['latency_ms']:.1f} ms")

    print("\n  FULL PROBABILITIES:")
    for lbl, prob in result["probabilities"].items():
        bar    = "█" * int(prob * 40)
        colour = label_colour if lbl == label else ""
        print(f"    {colour}{lbl:<35}{reset} {prob:.4f}  {bar}")

    print("─" * 70 + "\n")


def run_smoke_test(pipeline: Layer3Pipeline) -> None:
    print("\n" + "=" * 70)
    print("  LAYER 3 SMOKE TEST — 10 examples")
    print("  Pass condition: expected label in top_labels (not just top-1)")
    print("  (benign examples use risk_score ≤ 50 — NLI correctly bypassed)")
    print("=" * 70)

    passed = 0
    for desc, text, expected, risk_score in SMOKE_EXAMPLES:
        result     = pipeline.run(text, layer2_risk_score=risk_score)
        top_labels = result.get("top_labels", [])
        top_names  = [t["label"] for t in top_labels]

        # Pass if expected label appears anywhere in top_labels
        # (or equals primary label for benign fast-path)
        hit = (expected in top_names) or (result["label"] == expected)
        passed += hit
        status = "\033[92m✓\033[0m" if hit else "\033[91m✗\033[0m"
        gated  = " [L2 gated]" if risk_score <= pipeline.layer2_threshold else f" [L2 risk={risk_score}]"

        print(f"\n  {status}  {desc}{gated}")
        print(f"     primary  : {result['label']:<35} conf={result['confidence']:.3f}")

        if top_labels:
            for i, t in enumerate(top_labels):
                marker = "← expected" if t["label"] == expected else ""
                print(f"     top[{i}]   : {t['label']:<35} conf={t['confidence']:.3f}  {marker}")
        else:
            print(f"     top_labels: []  (benign fast-path)")

        print(f"     reason   : {result['reason']}")

    print(f"\n  Result: {passed}/{len(SMOKE_EXAMPLES)} passed")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3 NLI classifier — CLI demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-m", "--message",   type=str, help="Message text to classify")
    group.add_argument("--stdin",           action="store_true", help="Read from stdin")
    group.add_argument("--layer2-output",   type=str, help='JSON: {"text":"...","risk_score":75}')
    group.add_argument("--smoke-test",      action="store_true", help="Run 10 built-in examples")

    parser.add_argument("--risk-score", type=int, default=60,
                        help=(
                            "Simulated Layer 2 risk score (default: 60).\n"
                            "In the real pipeline this is set by Layer 2 LR.\n"
                            "Use --risk-score 10 to test benign fast-path."
                        ))
    parser.add_argument("--message-id",  type=str, default=None,
                        help="Optional message ID (passed through to Layer 4)")
    parser.add_argument("--timestamp",   type=str, default=None,
                        help="Optional ISO-8601 timestamp (passed through to Layer 4)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted display")
    args = parser.parse_args()

    # ── Resolve input ──────────────────────────────────────────────────────
    if args.smoke_test:
        pipeline = build_pipeline()
        run_smoke_test(pipeline)
        return

    if args.layer2_output:
        l2         = json.loads(args.layer2_output)
        text       = l2["text"]
        risk_score = l2.get("risk_score", 75)
        msg_id     = l2.get("message_id", args.message_id)
        ts         = l2.get("timestamp",  args.timestamp)
    elif args.stdin:
        text, risk_score, msg_id, ts = (
            sys.stdin.read().strip(), args.risk_score, args.message_id, args.timestamp
        )
    elif args.message:
        text, risk_score, msg_id, ts = (
            args.message, args.risk_score, args.message_id, args.timestamp
        )
    else:
        parser.print_help()
        sys.exit(1)

    pipeline = build_pipeline()
    result   = pipeline.run(
        text=text,
        layer2_risk_score=risk_score,
        message_id=msg_id,
        timestamp=ts,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        pretty_print(result, text=text)


if __name__ == "__main__":
    main()