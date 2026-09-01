"""
run_pipeline.py  (project root)
================================
CLI for the full hybrid SE detection pipeline.

Usage
-----
  # Single message (no conversation context)
  python run_pipeline.py -m "Urgent: verify your account now." --conv-id conv_01

  # Process a conversation from JSON file
  python run_pipeline.py --conv-file examples/conv_bec.json

  # Inline conversation (multiple -m flags share the same conv-id)
  python run_pipeline.py \\
      --conv-id conv_42 \\
      -m "Hi I'm from IT helpdesk" \\
      -m "We detected suspicious login on your account" \\
      -m "Please provide your credentials to verify"

  # JSON output
  python run_pipeline.py --conv-id conv_01 -m "Wire funds now" --json

Example conversation JSON file format:
  [
    {"text": "...", "message_id": "m1", "timestamp": "2026-05-08T09:00:00Z"},
    {"text": "...", "message_id": "m2", "timestamp": "2026-05-08T09:02:00Z"}
  ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE         = Path(__file__).resolve().parent
_LAYER3_DIR   = _HERE / "layer3_slm"
_LAYER4_DIR   = _HERE / "layer4"

for p in [str(_LAYER3_DIR), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Alert colours ─────────────────────────────────────────────────────────────
_ALERT_COLOUR = {
    "CRITICAL": "\033[91m",
    "HIGH":     "\033[91m",
    "MEDIUM":   "\033[93m",
    "LOW":      "\033[92m",
}
RESET = "\033[0m"


def _colour_alert(level: str) -> str:
    return f"{_ALERT_COLOUR.get(level, '')}{level}{RESET}"


def pretty_print_result(result: dict, msg_text: str | None = None) -> None:
    """Print full pipeline output in human-readable form."""
    last  = result.get("last_message", {})
    l4a   = result.get("l4a_context", {})

    print("\n" + "═" * 70)
    if msg_text:
        print(f"  TEXT          : {msg_text[:80]}{'…' if len(msg_text) > 80 else ''}")

    # Per-message (Layer 3)
    print(f"\n  ── Per-message (Layer 3) ──────────────────────────────────")
    print(f"  MSG ID        : {last.get('message_id', 'N/A')}")
    print(f"  TIMESTAMP     : {last.get('timestamp',  'N/A')}")
    print(f"  LABEL         : {last.get('label', '').upper()}")
    print(f"  CONFIDENCE    : {last.get('confidence', 0):.4f}")
    print(f"  L2 RISK       : {last.get('layer2_risk', 0)}")
    if last.get("top_labels"):
        print(f"  TOP LABELS    :")
        for t in last["top_labels"]:
            print(f"    {t['label']:<35} {t['confidence']:.4f}")
    print(f"  REASON        : {last.get('reason', '')}")

    # Conversation context (Layer 4a)
    print(f"\n  ── Conversation Context (Layer 4a) ───────────────────────")
    print(f"  CONV ID       : {l4a.get('conversation_id', 'N/A')}")
    print(f"  ACCUM RISK    : {l4a.get('accumulated_risk', 0):.3f}")
    print(f"  SUSPICIOUS    : {l4a.get('suspicious_flag', False)}")
    print(f"  SVM OVERRIDE  : {l4a.get('override_svm', False)}")
    print(f"  WINDOW MSGS   : {l4a.get('window_size', 0)}")

    # Conversation assessment (Layer 4b)
    alert = result.get("alert_level", "LOW")
    print(f"\n  ── Conversation Assessment (Layer 4b) ────────────────────")
    print(f"  ENTITY RISK   : {result.get('entity_risk', 0)}/100")
    print(f"  ALERT LEVEL   : {_colour_alert(alert)}")
    print(f"  ATTACK PATTERN: {result.get('attack_pattern', 'none')}")
    print(f"  DOMINANT LABEL: {result.get('dominant_label', 'none')}")
    print(f"  AVG CONFIDENCE: {result.get('confidence', 0):.4f}")

    reasons = result.get("reasons", [])
    if reasons:
        print(f"\n  REASONING TRAIL ({len(reasons)} messages):")
        for r in reasons:
            print(f"    {r}")

    print("═" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full hybrid SE detection pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m", "--message", action="append", dest="messages",
        help="Message text. Use multiple -m flags for a conversation.",
    )
    parser.add_argument("--conv-id",   default="conv_demo",
                        help="Conversation ID (default: conv_demo)")
    parser.add_argument("--conv-file", default=None,
                        help="JSON file with conversation messages")
    parser.add_argument("--json", action="store_true",
                        help="Print raw JSON output")
    args = parser.parse_args()

    if not args.messages and not args.conv_file:
        parser.print_help()
        sys.exit(1)

    # ── Build message list ────────────────────────────────────────────────────
    if args.conv_file:
        raw = json.loads(Path(args.conv_file).read_text())
        messages = [
            {
                "text":            m["text"],
                "conversation_id": m.get("conversation_id", args.conv_id),
                "message_id":      m.get("message_id"),
                "timestamp":       m.get("timestamp"),
            }
            for m in raw
        ]
    else:
        messages = [
            {
                "text":            text,
                "conversation_id": args.conv_id,
                "message_id":      f"{args.conv_id}_msg_{i+1:03d}",
                "timestamp":       None,
            }
            for i, text in enumerate(args.messages)
        ]

    # ── Load pipeline ────────────────────────────────────────────────────────
    from layer4.hybrid_pipeline import HybridPipeline
    pipe = HybridPipeline()

    # ── Process ──────────────────────────────────────────────────────────────
    results = pipe.process_conversation(messages)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for i, (msg, result) in enumerate(zip(messages, results)):
        print(f"\n  Message {i+1}/{len(messages)}")
        pretty_print_result(result, msg_text=msg["text"])


if __name__ == "__main__":
    main()