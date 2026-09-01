"""
layer4/test_layer4.py
======================
Phase 1 test suite for Layer 4a and Layer 4b using synthetic conversations.

Tests
-----
1. Risk counter accumulates and resets correctly
2. SVM override fires when conversation is suspicious
3. trust_build_then_exploit detected
4. authority_then_credential detected
5. bec_sequence detected
6. urgency_escalation detected
7. multi_vector detected
8. benign conversation stays LOW alert
9. Full pipeline smoke test (L4a → L3 → L4b)

Run from project root:
    python layer4/test_layer4.py
    python layer4/test_layer4.py --full   # includes full pipeline (loads L2 + L3 models)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_LAYER3_DIR   = _PROJECT_ROOT / "layer3_slm"

for p in [str(_LAYER3_DIR), str(_PROJECT_ROOT), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from layer4.layer4a_risk_counter    import RiskCounter, SUSPICIOUS_THRESHOLD
from layer4.layer4b_semantic_window import SemanticWindow

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Colour helpers ──────────────────────────────────────────────────────────
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
B = "\033[94m"   # blue
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {G}✓{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {R}✗{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Mock Layer 3 output builder
# ---------------------------------------------------------------------------

def _l3(
    label:       str,
    confidence:  float,
    message_id:  str = "msg_000",
    timestamp:   str = "2026-05-08T09:00:00Z",
    layer2_risk: int = 75,
    reason:      str = "Mock reason.",
) -> dict:
    return {
        "message_id":    message_id,
        "timestamp":     timestamp,
        "label":         label,
        "confidence":    confidence,
        "top_labels":    [{"label": label, "confidence": confidence}],
        "probabilities": {label: confidence},
        "reason":        reason,
        "layer2_risk":   layer2_risk,
        "latency_ms":    150.0,
    }


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        _ok(f"{name}")
    else:
        failed += 1
        _fail(f"{name}{(' — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Layer 4a tests
# ---------------------------------------------------------------------------

def test_risk_counter() -> None:
    print(f"\n{B}── Layer 4a: Risk Counter ──{RESET}")

    rc = RiskCounter()

    # T1: attack message increases accumulator
    ctx = rc.update("conv_1", "suspicious", 80)
    check("Attack message increases accumulator",
          ctx["accumulated_risk"] > 0,
          f"got {ctx['accumulated_risk']}")

    # T2: benign message decreases accumulator
    ctx2 = rc.update("conv_1", "benign", 5)
    check("Benign message decreases accumulator",
          ctx2["accumulated_risk"] < ctx["accumulated_risk"],
          f"before={ctx['accumulated_risk']:.3f} after={ctx2['accumulated_risk']:.3f}")

    # T3: accumulator never goes below 0
    rc2 = RiskCounter()
    for _ in range(10):
        rc2.update("conv_floor", "benign", 5)
    ctx3 = rc2.get_context("conv_floor")
    check("Accumulator floor is 0",
          ctx3["accumulated_risk"] >= 0,
          f"got {ctx3['accumulated_risk']}")

    # T4: suspicious_flag triggers after enough attacks
    rc3 = RiskCounter()
    for i in range(5):
        ctx4 = rc3.update("conv_sus", "suspicious", 90)
    check("suspicious_flag=True after high-risk messages",
          ctx4["suspicious_flag"],
          f"accumulated={ctx4['accumulated_risk']:.3f}")

    # T5: SVM override fires when conv is suspicious but msg is benign
    rc4 = RiskCounter()
    for _ in range(5):
        rc4.update("conv_override", "suspicious", 85)
    ctx5 = rc4.update("conv_override", "benign", 10)
    check("SVM override fires on benign msg in suspicious conv",
          ctx5["override_svm"],
          f"accumulated={ctx5['accumulated_risk']:.3f}")

    # T6: separate conversations tracked independently
    rc5 = RiskCounter()
    rc5.update("conv_A", "suspicious", 90)
    rc5.update("conv_A", "suspicious", 90)
    rc5.update("conv_B", "benign",     5)
    ctxA = rc5.get_context("conv_A")
    ctxB = rc5.get_context("conv_B")
    check("Conversations tracked independently",
          ctxA["accumulated_risk"] != ctxB["accumulated_risk"],
          f"A={ctxA['accumulated_risk']:.3f} B={ctxB['accumulated_risk']:.3f}")

    # T7: reset clears state
    rc5.reset("conv_A")
    check("reset() clears conversation state",
          rc5.get_context("conv_A") is None)

    # T8: recommended_min_conf is lower when suspicious
    rc6 = RiskCounter()
    for _ in range(5):
        rc6.update("conv_conf", "suspicious", 90)
    ctx6 = rc6.get_context("conv_conf")
    check("recommended_min_conf lowers when suspicious",
          ctx6["recommended_min_conf"] < 0.25,
          f"got {ctx6['recommended_min_conf']}")


# ---------------------------------------------------------------------------
# Layer 4b tests
# ---------------------------------------------------------------------------

def test_semantic_window() -> None:
    print(f"\n{B}── Layer 4b: Semantic Window ──{RESET}")

    # T1: trust_build_then_exploit
    sw = SemanticWindow()
    sw.update("conv_trust", _l3("benign",               0.88, "m1", "2026-05-08T09:00:00Z", 5))
    sw.update("conv_trust", _l3("benign",               0.91, "m2", "2026-05-08T09:02:00Z", 8))
    sw.update("conv_trust", _l3("pretexting",           0.72, "m3", "2026-05-08T09:04:00Z", 78))
    result = sw.update("conv_trust", _l3("credential_harvesting", 0.93, "m4", "2026-05-08T09:06:00Z", 91))
    check("trust_build_then_exploit detected",
          result["attack_pattern"] == "trust_build_then_exploit",
          f"got '{result['attack_pattern']}'")
    check("trust_build_then_exploit → HIGH/CRITICAL alert",
          result["alert_level"] in ("HIGH", "CRITICAL"),
          f"got '{result['alert_level']}'")

    # T2: authority_then_credential
    sw2 = SemanticWindow()
    sw2.update("conv_auth", _l3("pretexting",           0.78, "m1", layer2_risk=84))
    sw2.update("conv_auth", _l3("benign",               0.62, "m2", layer2_risk=12))
    result2 = sw2.update("conv_auth", _l3("credential_harvesting", 0.90, "m3", layer2_risk=88))
    check("authority_then_credential detected",
          result2["attack_pattern"] == "authority_then_credential",
          f"got '{result2['attack_pattern']}'")

    # T3: bec_sequence
    sw3 = SemanticWindow()
    sw3.update("conv_bec", _l3("spear_phishing",         0.71, "m1", layer2_risk=76))
    sw3.update("conv_bec", _l3("pretexting",             0.68, "m2", layer2_risk=72))
    result3 = sw3.update("conv_bec", _l3("business_email_compromise", 0.85, "m3", layer2_risk=89))
    check("bec_sequence detected",
          result3["attack_pattern"] == "bec_sequence",
          f"got '{result3['attack_pattern']}'")

    # T4: urgency_escalation
    sw4 = SemanticWindow()
    sw4.update("conv_esc", _l3("phishing", 0.30, "m1", layer2_risk=55))
    sw4.update("conv_esc", _l3("phishing", 0.55, "m2", layer2_risk=70))
    result4 = sw4.update("conv_esc", _l3("phishing", 0.85, "m3", layer2_risk=88))
    check("urgency_escalation detected",
          result4["attack_pattern"] == "urgency_escalation",
          f"got '{result4['attack_pattern']}'")

    # T5: multi_vector — needs 3+ distinct attack types with NO authority→credential
    # sequence that would trigger authority_then_credential first.
    # Using phishing, baiting, business_email_compromise, credential_harvesting:
    # none of the authority labels (pretexting/vishing/spear_phishing) appear
    # before credential_harvesting, so authority_then_credential won't fire.
    sw5 = SemanticWindow()
    sw5.update("conv_mv", _l3("phishing",                 0.72, "m1", layer2_risk=75))
    sw5.update("conv_mv", _l3("baiting",                  0.68, "m2", layer2_risk=71))
    sw5.update("conv_mv", _l3("business_email_compromise",0.61, "m3", layer2_risk=65))
    result5 = sw5.update("conv_mv", _l3("credential_harvesting", 0.80, "m4", layer2_risk=82))
    check("multi_vector detected",
          result5["attack_pattern"] == "multi_vector",
          f"got '{result5['attack_pattern']}'")

    # T6: benign conversation stays LOW
    sw6 = SemanticWindow()
    sw6.update("conv_clean", _l3("benign", 0.95, "m1", layer2_risk=5))
    sw6.update("conv_clean", _l3("benign", 0.92, "m2", layer2_risk=8))
    result6 = sw6.update("conv_clean", _l3("benign", 0.91, "m3", layer2_risk=6))
    check("All-benign conversation stays LOW alert",
          result6["alert_level"] == "LOW",
          f"got '{result6['alert_level']}'")

    # T7: entity_risk increases as more attacks arrive
    sw7 = SemanticWindow()
    r1 = sw7.update("conv_grow", _l3("phishing", 0.65, "m1", layer2_risk=70))
    r2 = sw7.update("conv_grow", _l3("phishing", 0.78, "m2", layer2_risk=82))
    r3 = sw7.update("conv_grow", _l3("phishing", 0.91, "m3", layer2_risk=92))
    check("entity_risk grows as attacks accumulate",
          r3["entity_risk"] >= r1["entity_risk"],
          f"r1={r1['entity_risk']} r3={r3['entity_risk']}")

    # T8: reasons trail has correct length
    check("reasons trail matches window size",
          len(result["reasons"]) == 4,
          f"got {len(result['reasons'])}")

    # T9: reset clears window
    sw7.reset("conv_grow")
    r_after = sw7.assess("conv_grow")
    check("reset() clears semantic window",
          r_after["entity_risk"] == 0)


# ---------------------------------------------------------------------------
# Full pipeline test (optional — requires L2 + L3 models)
# ---------------------------------------------------------------------------

def test_full_pipeline() -> None:
    print(f"\n{B}── Full Pipeline: L1 → L2 → L4a → L3 → L4b ──{RESET}")

    try:
        from layer4.hybrid_pipeline import HybridPipeline
        pipe = HybridPipeline()
    except Exception as e:
        print(f"  {Y}SKIP{RESET}  Could not load pipeline: {e}")
        return

    # Classic IT-support pretexting → credential_harvesting sequence
    conv = [
        {
            "text":            "Hi, this is the IT helpdesk. We have detected unusual activity.",
            "conversation_id": "test_conv_01",
            "message_id":      "m1",
            "timestamp":       "2026-05-08T09:00:00Z",
        },
        {
            "text":            "Please don't worry, we are just running a routine security check.",
            "conversation_id": "test_conv_01",
            "message_id":      "m2",
            "timestamp":       "2026-05-08T09:02:00Z",
        },
        {
            "text":            "To complete the verification, we will need your login credentials.",
            "conversation_id": "test_conv_01",
            "message_id":      "m3",
            "timestamp":       "2026-05-08T09:04:00Z",
        },
    ]

    results = pipe.process_conversation(conv)
    final   = results[-1]

    check("Pipeline processes full conversation without error",
          len(results) == 3)
    check("Final alert is HIGH or CRITICAL for IT-support scam",
          final["alert_level"] in ("HIGH", "CRITICAL"),
          f"got '{final['alert_level']}'")
    check("Attack pattern detected",
          final["attack_pattern"] not in ("no_attack_detected", "LOW"),
          f"got '{final['attack_pattern']}'")
    check("Reasons trail generated",
          len(final["reasons"]) > 0)

    print(f"\n  {B}Final assessment:{RESET}")
    print(f"    entity_risk    : {final['entity_risk']}")
    print(f"    alert_level    : {final['alert_level']}")
    print(f"    attack_pattern : {final['attack_pattern']}")
    print(f"    dominant_label : {final['dominant_label']}")
    print()
    for r in final["reasons"]:
        print(f"    {r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Include full pipeline test (loads L2 + L3 models)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LAYER 4 TEST SUITE")
    print("=" * 60)

    test_risk_counter()
    test_semantic_window()

    if args.full:
        test_full_pipeline()

    print("\n" + "=" * 60)
    colour = G if failed == 0 else R
    print(f"  {colour}{passed} passed  {failed} failed{RESET}")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()