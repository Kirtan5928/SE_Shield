import streamlit as st
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SE Detection System",
    page_icon   = "🛡️",          # FIX: plain string "shield" is not a valid icon
    layout      = "wide",
    initial_sidebar_state = "collapsed"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Input */
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 14px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #1f6feb;
        border-color: #1f6feb;
        color: white;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #388bfd;
    }

    /* Cards */
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* Result badge */
    .badge-attack {
        display: inline-block;
        background-color: #3d1f1f;
        border: 1px solid #f85149;
        color: #f85149;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 1px;
    }
    .badge-benign {
        display: inline-block;
        background-color: #1a2e1a;
        border: 1px solid #3fb950;
        color: #3fb950;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 1px;
    }

    /* Score display */
    .score-number {
        font-size: 52px;
        font-weight: 800;
        line-height: 1;
        margin: 8px 0;
    }
    .score-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk bar container */
    .risk-bar-container {
        background-color: #21262d;
        border-radius: 8px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 8px 0;
    }

    /* Pipeline step */
    .pipeline-step {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
    }
    .pipeline-step-active {
        border-color: #1f6feb;
        background-color: #1a2433;
    }
    .pipeline-step-bypassed {
        border-color: #3fb950;
        background-color: #1a2e1a;
        color: #8b949e;
    }

    /* Metric row */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #21262d;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-key { color: #8b949e; font-size: 13px; }
    .metric-val { color: #e6edf3; font-size: 13px; font-weight: 600; }

    /* Section header */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #21262d;
    }

    /* Divider */
    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# FIX: dirname(dirname(...)) goes up from app/ to hybrid_se/ where models/ lives
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource(show_spinner=False)
def load_models():
    pipeline   = joblib.load(os.path.join(BASE_DIR, 'models', 'final_pipeline.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
    return pipeline, vectorizer

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def compute_confidence(probability: float, threshold: float = 0.5) -> dict:
    """
    Two confidence measures:
    1. Distance: |P - threshold| * 200  (range 0-100)
    2. Entropy : (1 - H(p)/log2(2)) * 100
    """
    eps           = 1e-10
    distance_conf = min(abs(probability - threshold) * 200, 100)

    p             = np.clip(probability, eps, 1 - eps)
    ent           = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    entropy_conf  = max(0, (1 - ent) * 100)

    return {
        'distance' : round(distance_conf, 1),
        'entropy'  : round(entropy_conf,  1),
    }


def predict_pipeline(text: str, pipeline: dict, vectorizer) -> dict:
    """
    Hybrid SVM → LR pipeline with soft routing (confidence-based).
    """

    if not text or not text.strip():
        return None

    svm_model = pipeline['svm_model']
    lr_model  = pipeline['lr_model']
    threshold = pipeline.get('threshold', 0.5)

    # Vectorize input
    X = vectorizer.transform([text])

    # ── Stage 1A: SVM (probabilistic gatekeeper) ──────────────────────────────
    try:
        svm_prob = svm_model.predict_proba(X)[0][1]
    except Exception:
        # fallback if probability not available (should not happen in your case)
        decision = svm_model.decision_function(X)[0]
        svm_prob = 1 / (1 + np.exp(-decision))  # sigmoid approximation

    # 🔥 Soft boundary: only exit if VERY confident benign
    if svm_prob < 0.2:
        return {
            'prediction'  : 'BENIGN',
            'risk_score'  : round(svm_prob * 100, 1),
            'probability' : round(svm_prob, 4),
            'confidence'  : compute_confidence(svm_prob, threshold),
            'svm_pred'    : 'BENIGN',
            'svm_prob'    : round(svm_prob * 100, 1),
            'lr_ran'      : False,
            'lr_prob'     : None,
        }

    # ── Stage 1B: Logistic Regression (risk scorer) ───────────────────────────
    lr_prob = lr_model.predict_proba(X)[0][1]

    risk_score = round(lr_prob * 100, 1)
    prediction = 'ATTACK' if lr_prob >= threshold else 'BENIGN'

    return {
        'prediction'  : prediction,
        'risk_score'  : risk_score,
        'probability' : round(lr_prob, 4),
        'confidence'  : compute_confidence(lr_prob, threshold),
        'svm_pred'    : 'ATTACK',
        'svm_prob'    : round(svm_prob * 100, 1),
        'lr_ran'      : True,
        'lr_prob'     : round(lr_prob * 100, 1),
    }
# ═══════════════════════════════════════════════════════════════════════════════
# RISK BAND HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def get_risk_color(score: float) -> str:
    if score < 25:  return '#3fb950'
    if score < 50:  return '#d29922'
    if score < 75:  return '#f0883e'
    return '#f85149'

def get_risk_label(score: float) -> str:
    if score < 25:  return 'LOW'
    if score < 50:  return 'MEDIUM'
    if score < 75:  return 'HIGH'
    return 'CRITICAL'

def risk_bar_html(score: float) -> str:
    color = get_risk_color(score)
    pct   = min(score, 100)
    return f"""
    <div class="risk-bar-container">
      <div style="
        width: {pct}%;
        height: 100%;
        background: linear-gradient(90deg, {color}99, {color});
        border-radius: 8px;
        transition: width 0.4s ease;
      "></div>
    </div>"""

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
EXAMPLES = {
    "Phishing Message": (
        "URGENT: Your bank account has been suspended due to suspicious activity. "
        "Verify your credentials immediately at http://secure-login-update.com "
        "to avoid permanent account closure. Failure to act within 24 hours will "
        "result in loss of access. Click here now and provide your password."
    ),
    "Normal Message": (
        "Hi, just wanted to follow up on the meeting we had last Thursday. "
        "I've attached the project summary document for your review. "
        "Let me know if you have any questions or need any clarifications. "
        "Looking forward to hearing your thoughts. Best regards."
    ),
    "Borderline Message": (
        "Hello, this is a reminder that your subscription is expiring soon. "
        "Please log in to your account to review your plan options. "
        "Contact our support team if you need any assistance."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Load models
    with st.spinner("Loading models..."):
        try:
            pipeline, vectorizer = load_models()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.info(f"Looking in: {os.path.join(BASE_DIR, 'models')}")
            return

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:6px;">
            HYBRID ML PIPELINE — STAGE 1
        </div>
        <h1 style="font-size:26px; font-weight:800; color:#e6edf3; margin:0;">
            Social Engineering Detection System
        </h1>
        <p style="color:#8b949e; font-size:14px; margin:6px 0 0 0;">
            SVM gatekeeper + Logistic Regression risk scorer
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Layout: input col + result col ────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            '<div class="section-header">Input</div>',
            unsafe_allow_html=True
        )

        # Initialise session state
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ''

        input_text = st.text_area(
            label            = "Message or email text",
            value            = st.session_state.input_text,
            height           = 200,
            placeholder      = "Paste an email or message here...",
            label_visibility = "collapsed"
        )

        # Analyze button
        analyze_clicked = st.button(
            "Analyze", type="primary", use_container_width=True
        )

        # Demo examples
        st.markdown(
            '<div class="section-header" style="margin-top:20px;">Examples</div>',
            unsafe_allow_html=True
        )

        ex_cols = st.columns(3)
        for i, (label, text) in enumerate(EXAMPLES.items()):
            with ex_cols[i]:
                if st.button(label, use_container_width=True):
                    st.session_state.input_text = text
                    st.rerun()

        # Architecture info card
        st.markdown("""
        <div class="card" style="margin-top:20px;">
            <div class="section-header">Architecture</div>
            <div class="metric-row">
                <span class="metric-key">Stage 1A</span>
                <span class="metric-val">SVM — binary gatekeeper</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Stage 1B</span>
                <span class="metric-val">LR — risk scorer (if flagged)</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Features</span>
                <span class="metric-val">TF-IDF (10,000 features)</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">SVM Accuracy</span>
                <span class="metric-val">98.95%</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">LR Accuracy</span>
                <span class="metric-val">98.93%</span>
            </div>
            <div class="metric-row">
                <span class="metric-key">Threshold</span>
                <span class="metric-val">0.5</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Result column ──────────────────────────────────────────────────────────
    with col_result:
        st.markdown(
            '<div class="section-header">Analysis Result</div>',
            unsafe_allow_html=True
        )

        # Run prediction
        result = None
        if analyze_clicked and input_text.strip():
            result = predict_pipeline(input_text, pipeline, vectorizer)
        elif analyze_clicked and not input_text.strip():
            st.warning("Please enter a message to analyze.")

        if result is None:
            # Placeholder state
            st.markdown("""
            <div class="card" style="text-align:center; padding:60px 24px;">
                <div style="font-size:40px; margin-bottom:12px; opacity:0.3;">🛡️</div>
                <div style="color:#8b949e; font-size:14px;">
                    Enter a message and click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            score      = result['risk_score']
            pred       = result['prediction']
            prob       = result['probability']
            conf       = result['confidence']
            risk_color = get_risk_color(score)
            risk_label = get_risk_label(score)

            # ── Main result card ───────────────────────────────────────────────
            badge_class = 'badge-attack' if pred == 'ATTACK' else 'badge-benign'
            st.markdown(f"""
            <div class="card">
                <div class="section-header">Verdict</div>
                <div style="display:flex; align-items:center;
                            justify-content:space-between; flex-wrap:wrap; gap:12px;">
                    <div>
                        <span class="{badge_class}">{pred}</span>
                        <div style="margin-top:8px; color:#8b949e; font-size:12px;">
                            Risk Band: <span style="color:{risk_color};
                            font-weight:700;">{risk_label}</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div class="score-label">Risk Score</div>
                        <div class="score-number"
                             style="color:{risk_color};">{score:.0f}</div>
                        <div style="font-size:11px; color:#8b949e;">out of 100</div>
                    </div>
                </div>
                <div style="margin-top:16px;">
                    {risk_bar_html(score)}
                    <div style="display:flex; justify-content:space-between;
                                font-size:10px; color:#8b949e; margin-top:4px;">
                        <span>0 — Low</span>
                        <span>25</span>
                        <span>50</span>
                        <span>75</span>
                        <span>100 — Critical</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence card ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="card">
                <div class="section-header">Confidence</div>
                <div class="metric-row">
                    <span class="metric-key">
                        Distance Confidence
                        <span style="font-size:10px; color:#58a6ff;">
                         — |P - 0.5| x 200
                        </span>
                    </span>
                    <span class="metric-val">{conf['distance']:.1f} / 100</span>
                </div>
                <div style="background:#21262d; border-radius:6px;
                            height:6px; margin:4px 0 12px 0; overflow:hidden;">
                    <div style="width:{conf['distance']}%; height:100%;
                                background:#58a6ff; border-radius:6px;"></div>
                </div>
                <div class="metric-row">
                    <span class="metric-key">
                        Entropy Confidence
                        <span style="font-size:10px; color:#58a6ff;">
                         — 1 - H(p)
                        </span>
                    </span>
                    <span class="metric-val">{conf['entropy']:.1f} / 100</span>
                </div>
                <div style="background:#21262d; border-radius:6px;
                            height:6px; margin:4px 0 8px 0; overflow:hidden;">
                    <div style="width:{conf['entropy']}%; height:100%;
                                background:#58a6ff; border-radius:6px;"></div>
                </div>
                <div style="font-size:11px; color:#8b949e; margin-top:8px;">
                    Higher score = model is more certain of this prediction.
                    Scores below 20 indicate borderline cases that may warrant review.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Pipeline trace card ────────────────────────────────────────────
            st.markdown(
                '<div class="section-header">Pipeline Trace</div>',
                unsafe_allow_html=True
            )

            # Stage 1A
            svm_class = 'pipeline-step-active' if result['lr_ran'] else 'pipeline-step-bypassed'

            # FIX: f-string formatting expression split out to avoid ambiguous :.1f inside nested {}
            benign_conf = round(100 - result['svm_prob'], 1)
            svm_status  = (
                f"Flagged as suspicious — {result['svm_prob']}% attack probability"
                if result['lr_ran']
                else f"Classified as benign — {benign_conf}% benign confidence"
            )

            st.markdown(f"""
            <div class="pipeline-step {svm_class}">
                <span style="color:#8b949e; font-size:11px;">STAGE 1A</span>
                <span style="margin-left:8px; font-weight:600;">SVM Gatekeeper</span>
                <div style="margin-top:4px; font-size:12px; color:#8b949e;">
                    {svm_status}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Stage 1B
            if result['lr_ran']:
                st.markdown(f"""
                <div class="pipeline-step pipeline-step-active">
                    <span style="color:#8b949e; font-size:11px;">STAGE 1B</span>
                    <span style="margin-left:8px; font-weight:600;">LR Risk Scorer</span>
                    <div style="margin-top:4px; font-size:12px; color:#8b949e;">
                        P(attack) = {prob:.4f}
                        — Risk Score = {prob:.4f} x 100 = <strong
                        style="color:{risk_color};">{score:.1f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pipeline-step pipeline-step-bypassed">
                    <span style="color:#8b949e; font-size:11px;">STAGE 1B</span>
                    <span style="margin-left:8px;">LR Risk Scorer</span>
                    <div style="margin-top:4px; font-size:12px; color:#6e7681;">
                        Skipped — message cleared at Stage 1A
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Output
            pred_color = '#f85149' if pred == 'ATTACK' else '#3fb950'
            st.markdown(f"""
            <div class="pipeline-step" style="border-color:#30363d; margin-top:4px;">
                <span style="color:#8b949e; font-size:11px;">OUTPUT</span>
                <span style="margin-left:8px; font-weight:600;">Final Verdict</span>
                <div style="margin-top:4px; font-size:12px; color:#8b949e;">
                    Label = <strong style="color:{pred_color};">{pred}</strong>
                    &nbsp;|&nbsp;
                    Risk Score = <strong style="color:{risk_color};">{score:.0f}</strong>
                    &nbsp;|&nbsp;
                    Band = <strong style="color:{risk_color};">{risk_label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between;
                align-items:center; padding:8px 0; color:#8b949e; font-size:11px;">
        <span>Hybrid SE Detection System — Stage 1 Pipeline</span>
        <span>SVM + Logistic Regression + TF-IDF (10k features)</span>
        <span>Dataset: Enron + Phishing + Spam + Synthetic — 140k rows</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()