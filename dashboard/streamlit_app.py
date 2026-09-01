"""
dashboard/streamlit_app.py
===========================
SE Shield — Analyst Dashboard.
Talks to the FastAPI server at localhost:8000.

Run:
  streamlit run dashboard/streamlit_app.py
"""

import requests
import streamlit as st

SERVER = "http://127.0.0.1:8000"

# ── Preset examples for presentation ─────────────────────────────────────────
PRESETS = {
    "⚠️  Attack — IT Support Pretexting (3-msg sequence)": [
        {
            "label": "Message 1 of 3",
            "text":  "Hi, this is Sarah from the IT helpdesk. We have detected unusual login activity on your account from an unrecognised device.",
            "tag":   "pretexting",
        },
        {
            "label": "Message 2 of 3",
            "text":  "This is just a routine security check, nothing to worry about. We carry out these verifications regularly to keep your account safe.",
            "tag":   "trust_build",
        },
        {
            "label": "Message 3 of 3",
            "text":  "To complete the verification and restore full access, we will need you to confirm your username and current password through our secure portal.",
            "tag":   "exploit",
        },
    ],
    "⚠️  Attack — BEC Invoice Fraud (single)": [
        {
            "label": "BEC message",
            "text":  "Hi, following up on our earlier conversation regarding the pending invoice. Could you please update the bank account details on file and process the transfer of $47,500 before close of business today? The CEO has approved this — it is time sensitive.",
            "tag":   "bec",
        },
    ],
    "✅  Benign — Workplace Message": [
        {
            "label": "Benign message",
            "text":  "Hey, just checking in on the project timeline. Can we sync up tomorrow around 2pm to go over the Q3 deliverables? I will send a calendar invite.",
            "tag":   "benign",
        },
    ],
    "🔶  Borderline — Ambiguous Urgency": [
        {
            "label": "Borderline message",
            "text":  "Your subscription is expiring soon. Please log in to your account and update your payment information to avoid any interruption to your service.",
            "tag":   "borderline",
        },
    ],
}

PRESET_CONV_IDS = {
    "⚠️  Attack — IT Support Pretexting (3-msg sequence)": "demo_it_support",
    "⚠️  Attack — BEC Invoice Fraud (single)":             "demo_bec",
    "✅  Benign — Workplace Message":                       "demo_benign",
    "🔶  Borderline — Ambiguous Urgency":                   "demo_borderline",
}

st.set_page_config(
    page_title="SE Shield",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07101C !important;
    color: #A8CCDE !important;
}
.stApp { background-color: #07101C !important; }
[data-testid="stSidebar"] {
    background-color: #0C1929 !important;
    border-right: 1px solid #1A2E42 !important;
}
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #FFFFFF !important; }
.stButton > button {
    background-color: #00C9A7 !important;
    color: #07101C !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    letter-spacing: 1px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextArea textarea, .stTextInput input {
    background-color: #122034 !important;
    color: #FFFFFF !important;
    border: 1px solid #1A2E42 !important;
}
div[data-baseweb="select"] > div {
    background-color: #122034 !important;
    border-color: #1A2E42 !important;
    color: #FFFFFF !important;
}
.metric-card {
    background: #122034; border: 1px solid #1A2E42;
    border-radius: 8px; padding: 16px; text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem; font-weight: 700; line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem; color: #5E8095;
    letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px;
}
.alert-box {
    border-left: 4px solid; padding: 12px 16px;
    border-radius: 0 6px 6px 0; background: #0C1929; margin-bottom: 16px;
}
.reason-row {
    padding: 8px 12px; margin-bottom: 5px; border-radius: 5px;
    background: #122034; border-left: 3px solid #5E8095;
    font-size: 0.82rem; color: #A8CCDE;
}
.reason-row.attack { border-left-color: #E05252; }
.reason-row.benign { border-left-color: #3DC97A; }
.risk-track { height: 10px; border-radius: 5px; background: #1A2E42; overflow: hidden; margin: 6px 0; }
.risk-fill  { height: 100%; border-radius: 5px; }
.preset-card {
    background: #0C1929; border: 1px solid #1A2E42; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px; font-size: 0.82rem; color: #A8CCDE;
    line-height: 1.55;
}
.preset-tag-attack    { color: #E05252; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
.preset-tag-benign    { color: #3DC97A; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
.preset-tag-borderline{ color: #F0943A; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


def server_health():
    try:
        r = requests.get(f"{SERVER}/health", timeout=3)
        return r.ok, (r.json() if r.ok else {})
    except Exception:
        return False, {}


def scan_message(text, conv_id, msg_id=None, timestamp=None,
                 sender_name=None, platform="manual"):
    payload = {
        "text":            text,
        "conversation_id": conv_id,
        "message_id":      msg_id,
        "timestamp":       timestamp,
        "sender_name":     sender_name,
        "platform":        platform,
    }
    r = requests.post(f"{SERVER}/scan", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def reset_conv(conv_id):
    try:
        requests.delete(f"{SERVER}/conversation/{conv_id}", timeout=5)
    except Exception:
        pass


def alert_colour(level):
    return {"CRITICAL": "#E05252", "HIGH": "#E05252",
            "MEDIUM": "#F0943A", "LOW": "#3DC97A"}.get(level, "#5E8095")


def risk_colour(risk):
    return "#E05252" if risk >= 60 else ("#F0943A" if risk >= 35 else "#3DC97A")


def render_result(latest, scanned_text):
    level   = latest.get("alert_level", "LOW")
    risk    = latest.get("entity_risk", 0)
    pattern = latest.get("attack_pattern", "none")
    dom_lbl = latest.get("dominant_label", "none")
    conf    = latest.get("confidence", 0.0)
    reasons = latest.get("reasons", [])
    win_sz  = latest.get("window_size", 0)
    last    = latest.get("last_message", {})
    ac      = alert_colour(level)
    rc      = risk_colour(risk)

    # Alert banner
    st.markdown(f"""
    <div class="alert-box" style="border-color:{ac}">
        <span style="font-family:'Space Mono',monospace;font-size:1.4rem;font-weight:700;color:{ac}">{level}</span>
        &nbsp;&nbsp;
        <span style="font-size:0.85rem;color:#A8CCDE;text-transform:uppercase;letter-spacing:1px">
            {pattern.replace('_', ' ')}
        </span>
    </div>""", unsafe_allow_html=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{rc}">{risk}</div><div class="metric-label">Entity Risk / 100</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#FFF">{win_sz}</div><div class="metric-label">Messages in Window</div></div>', unsafe_allow_html=True)
    with m3:
        lbl_d = dom_lbl.replace("_", " ").upper()
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1rem;color:{ac};padding:10px 0">{lbl_d}</div><div class="metric-label">Dominant Attack Type</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#3FA0E0">{conf:.2f}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)

    # Risk bar
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-top:16px">
        <span style="font-size:0.72rem;color:#5E8095;letter-spacing:1px;width:90px">RISK LEVEL</span>
        <div class="risk-track" style="flex:1"><div class="risk-fill" style="width:{risk}%;background:{rc}"></div></div>
        <span style="font-family:'Space Mono',monospace;font-size:0.85rem;color:{rc};width:50px">{risk}/100</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_detail, col_reasons = st.columns([2, 3])

    with col_detail:
        st.markdown("**Last Message**")
        if last:
            st.markdown(f"""
            <div style="background:#122034;border:1px solid #1A2E42;border-radius:8px;padding:14px">
                <div style="font-size:0.7rem;color:#5E8095;letter-spacing:1px">SCANNED TEXT</div>
                <div style="font-size:0.82rem;color:#A8CCDE;margin-bottom:10px;font-style:italic">"{scanned_text[:120]}..."</div>
                <div style="font-size:0.7rem;color:#5E8095;letter-spacing:1px">LABEL</div>
                <div style="font-size:1rem;font-weight:600;color:#FFF;margin-bottom:10px">{last.get('label','').replace('_',' ').upper()}</div>
                <div style="font-size:0.7rem;color:#5E8095;letter-spacing:1px">CONFIDENCE</div>
                <div style="font-size:0.95rem;color:#3FA0E0;margin-bottom:10px">{last.get('confidence', 0):.4f}</div>
                <div style="font-size:0.7rem;color:#5E8095;letter-spacing:1px">L2 RISK SCORE</div>
                <div style="font-size:0.9rem;color:#A8CCDE;margin-bottom:10px">{last.get('layer2_risk', 0)}/100</div>
                <div style="font-size:0.7rem;color:#5E8095;letter-spacing:1px">REASON</div>
                <div style="font-size:0.8rem;color:#A8CCDE;line-height:1.5">{str(last.get('reason',''))[:220]}</div>
            </div>""", unsafe_allow_html=True)

    with col_reasons:
        st.markdown("**Reasoning Trail**")
        if reasons:
            html = ""
            for r in reasons:
                cls = "benign" if "benign" in r.lower() else "attack"
                html += f'<div class="reason-row {cls}">{r}</div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No trail yet.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡 SE SHIELD")
    st.markdown("---")

    ok, health_data = server_health()
    if ok:
        active = health_data.get("active_conversations", 0)
        st.success(f"Server online · {active} active conversation{'s' if active != 1 else ''}")
    else:
        st.error("Server offline")
        st.code("uvicorn server.main:app --host 127.0.0.1 --port 8000")
        st.stop()

    st.markdown("---")
    st.markdown("**DEMO EXAMPLES**")
    st.caption("Ready-to-run examples for presentation")

    selected_preset = st.selectbox(
        "Select example",
        options=["— Custom message —"] + list(PRESETS.keys()),
        label_visibility="collapsed",
    )

    if selected_preset != "— Custom message —":
        msgs = PRESETS[selected_preset]
        pid  = PRESET_CONV_IDS[selected_preset]

        # Show preview of messages
        for m in msgs:
            tag = m["tag"]
            tag_class = "borderline" if tag in ("borderline", "trust_build") else \
                        "benign" if tag == "benign" else "attack"
            st.markdown(f"""
            <div class="preset-card">
                <div class="preset-tag-{tag_class}">{m['label'].upper()}</div>
                <div style="margin-top:4px">{m['text'][:100]}{'...' if len(m['text'])>100 else ''}</div>
            </div>""", unsafe_allow_html=True)

        col_run, col_rst = st.columns(2)
        with col_run:
            run_preset = st.button("▶ RUN DEMO", use_container_width=True)
        with col_rst:
            if st.button("↺ RESET", use_container_width=True):
                reset_conv(pid)
                st.session_state["preset_results"] = []
                st.session_state["preset_active"]  = None
                st.rerun()

        if run_preset:
            reset_conv(pid)
            st.session_state["preset_results"] = []
            st.session_state["preset_active"]  = selected_preset
            with st.spinner("Running demo sequence..."):
                collected = []
                for i, m in enumerate(msgs):
                    try:
                        result = scan_message(
                            text    = m["text"],
                            conv_id = pid,
                            msg_id  = f"{pid}_m{i+1:02d}",
                        )
                        collected.append({"text": m["text"], "label": m["label"], "result": result})
                    except Exception as e:
                        st.error(f"Scan failed on {m['label']}: {e}")
                        break
                st.session_state["preset_results"] = collected
            st.rerun()

    st.markdown("---")
    st.markdown("**Custom Scan**")
    conv_id = st.text_input(
        "Conversation ID",
        value=st.session_state.get("conv_id", "conv_demo"),
        label_visibility="collapsed",
        placeholder="conv_demo",
    )
    st.session_state["conv_id"] = conv_id

    if st.button("Reset Custom Window", use_container_width=True):
        reset_conv(conv_id)
        st.session_state["results"] = []
        st.success("Reset.")

    st.markdown("---")
    st.caption(
        "Each message adds context to the sliding window. "
        "Pattern detection improves as more messages are scanned."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# SE Shield — Analyst Dashboard")
st.markdown("---")

# ── PRESET RESULTS VIEW ───────────────────────────────────────────────────────
preset_results = st.session_state.get("preset_results", [])
preset_active  = st.session_state.get("preset_active", None)

if preset_results and preset_active and preset_active != "— Custom message —":
    final = preset_results[-1]["result"]
    level = final.get("alert_level", "LOW")
    ac    = alert_colour(level)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#5E8095;letter-spacing:2px">DEMO</span>
        <span style="font-size:1.1rem;font-weight:600;color:#FFF">{preset_active.split('—')[1].strip() if '—' in preset_active else preset_active}</span>
    </div>""", unsafe_allow_html=True)

    render_result(final, preset_results[-1]["text"])

    if len(preset_results) > 1:
        st.markdown("**Message-by-Message Breakdown**")
        for i, entry in enumerate(preset_results):
            r   = entry["result"]
            lvl = r.get("alert_level", "LOW")
            lbl = r.get("last_message", {}).get("label", "").replace("_", " ").upper()
            cf  = r.get("last_message", {}).get("confidence", 0)
            ris = r.get("entity_risk", 0)
            ac2 = alert_colour(lvl)
            with st.expander(
                f"Msg {i+1} — {entry['label']} → {lbl} | {lvl} | Risk {ris}/100",
                expanded=(i == len(preset_results)-1),
            ):
                st.markdown(f"> *{entry['text']}*")
                st.markdown(f"""
                <div style="display:flex;gap:24px;margin-top:8px">
                    <div><span style="font-size:0.7rem;color:#5E8095">ALERT</span><br>
                         <span style="color:{ac2};font-weight:700">{lvl}</span></div>
                    <div><span style="font-size:0.7rem;color:#5E8095">ENTITY RISK</span><br>
                         <span style="color:{ac2};font-weight:700">{ris}/100</span></div>
                    <div><span style="font-size:0.7rem;color:#5E8095">LABEL</span><br>
                         <span style="color:#FFF">{lbl}</span></div>
                    <div><span style="font-size:0.7rem;color:#5E8095">CONFIDENCE</span><br>
                         <span style="color:#3FA0E0">{cf:.3f}</span></div>
                </div>""", unsafe_allow_html=True)
                reason = r.get("last_message", {}).get("reason", "")
                if reason:
                    st.caption(f"Reason: {reason[:180]}")

    st.markdown("---")

# ── CUSTOM SCAN ───────────────────────────────────────────────────────────────
st.markdown("**Custom Message Scan**")
st.markdown(f"Conversation: `{conv_id}`")

col_input, col_btn = st.columns([5, 1])
with col_input:
    msg_text = st.text_area(
        "Message", placeholder="Paste or type the message to scan...",
        height=90, label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    do_scan = st.button("SCAN", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = []

if do_scan:
    if not msg_text.strip():
        st.warning("Enter a message first.")
    else:
        with st.spinner("Running detection..."):
            try:
                result = scan_message(text=msg_text.strip(), conv_id=conv_id)
                st.session_state["results"].append(
                    {"text": msg_text.strip(), "result": result}
                )
                st.session_state["preset_results"] = []
                st.session_state["preset_active"]  = None
            except Exception as e:
                st.error(f"Scan failed: {e}")

if st.session_state["results"] and not preset_results:
    latest = st.session_state["results"][-1]["result"]
    render_result(latest, st.session_state["results"][-1]["text"])

    if len(st.session_state["results"]) > 1:
        st.markdown("---")
        st.markdown("**Scan History**")
        for i, entry in enumerate(reversed(st.session_state["results"])):
            r   = entry["result"]
            lvl = r.get("alert_level", "LOW")
            pat = r.get("attack_pattern", "").replace("_", " ").upper()
            lbl = r.get("last_message", {}).get("label", "").replace("_", " ").upper()
            cf  = r.get("last_message", {}).get("confidence", 0)
            with st.expander(
                f"Msg {len(st.session_state['results'])-i} — {lbl} | {lvl}",
                expanded=(i == 0),
            ):
                st.markdown(f"> {entry['text'][:200]}")
                st.markdown(f"**Pattern:** {pat}  **Confidence:** {cf:.3f}  **Risk:** {r.get('entity_risk',0)}/100")

elif not preset_results:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#5E8095">
        <div style="font-size:2rem;margin-bottom:12px">🛡</div>
        <div style="font-size:1rem;margin-bottom:8px">Select a demo example from the sidebar or type a custom message above.</div>
    </div>""", unsafe_allow_html=True)