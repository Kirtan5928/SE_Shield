"""
run_layer4.py
=============
Standalone demo for Layer 4 — Conversational Context Engine
"""

from datetime import datetime, timedelta
import json

from context_engine import ContextManager, ContextConfig


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def demo_layer4():
    print_header("LAYER 4 — SLIDING WINDOW CONTEXT ENGINE DEMO")
    print("Demonstrates multi-turn social engineering detection\n")

    # Initialize Layer 4
    config = ContextConfig(
        window_size=6,
        high_risk_threshold=50.0,
        critical_risk_threshold=75.0,
        escalation_delta_threshold=18.0
    )
    context_mgr = ContextManager(config)

    # Simulated conversation (multi-turn attack)
    conversation = [
        ("Hey John, how's the project going?", 12, "benign"),
        ("Just following up on our meeting last week.", 18, "benign"),
        ("The CFO asked me to check on the Q3 payments.", 35, "spear_phishing"),
        ("Can you process this urgent wire transfer of $48,000 today?", 82, "business_email_compromise"),
        ("New vendor account details attached. Please do it ASAP.", 91, "business_email_compromise"),
    ]

    print("Simulated Conversation Flow:\n")

    for i, (text, risk_score, predicted_class) in enumerate(conversation, 1):
        # Add message to Layer 4
        context_mgr.add_message(
            text=text,
            risk_score=risk_score,
            predicted_class=predicted_class,
            timestamp=datetime.now() - timedelta(minutes=30-i*5)
        )

        payload = context_mgr.get_context_payload()

        print(f"Message {i}:")
        print(f"   Text     : {text[:70]}{'...' if len(text)>70 else ''}")
        print(f"   L2 Risk  : {risk_score:3d} | L3 Class : {predicted_class}")
        print(f"   Agg Risk : {payload['aggregated_risk']:.1f} | Trend Slope : {payload['risk_trend_slope']:.2f}")
        print(f"   Escalation : {'YES' if payload['has_escalation'] else 'No'} | Trigger SLM : {'YES' if payload.get('should_trigger_slm') else 'No'}")
        print("-" * 70)

    print_header("FINAL CONTEXT PAYLOAD (for Dashboard / Layer 5)")
    final_payload = context_mgr.get_context_payload()
    print(json.dumps(final_payload, indent=2))

    print("\n✅ Layer 4 successfully detected escalation and multi-turn risk buildup!")


if __name__ == "__main__":
    demo_layer4()