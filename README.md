# SE Shield — Hybrid Social Engineering Detection System

A multi-layer hybrid pipeline for detecting social engineering attacks across email, WhatsApp, and iMessage in real time. Combines classical ML, zero-shot NLI, and a sliding window context engine to detect both single-message and multi-step SE attacks.

---

## Architecture

```
Raw message + sender identity
        │
        ▼
Layer 1  Preprocessing       unicode normalise, homoglyph substitution
        │
        ▼
Layer 2  ML Triage           TF-IDF + SVM binary gate + LR risk score
        │
        ▼
Layer 4a Risk Counter        conversation accumulator, SVM override
        │
        ▼
Layer 3  NLI Engine          cross-encoder/nli-deberta-v3-small, 7 labels, ~150ms
        │
        ▼
Layer 4b Semantic Window     6 attack patterns, entity risk 0–100
        │
        ▼
Layer 5  Interfaces          FastAPI + Dashboard + Chrome Extension + Menu Bar App
```

---

## Features

- **Zero-shot NLI classification** — detects attack sub-types without any SE-labelled training data
- **7 attack labels** — phishing, spear phishing, pretexting, credential harvesting, baiting, vishing, BEC
- **Sliding window context** — detects multi-step attacks invisible to per-message classifiers
- **6 attack patterns** — `trust_build_then_exploit`, `authority_then_credential`, `bec_sequence`, `urgency_escalation`, `delayed_execution`, `multi_vector`
- **Cross-platform** — Gmail, Outlook, WhatsApp Web, macOS Messages, WhatsApp Desktop
- **Cross-platform identity resolution** — same attacker across platforms shares one conversation window
- **Local inference only** — no external API calls, all models run on-device

---

## Results

| Metric | Value |
|---|---|
| Recall (attack, integrated) | 0.9969 |
| Precision | 0.9932 |
| F1 | 0.9950 |
| Accuracy | 0.9936 |
| Latency p50 | 143ms |
| Latency p95 | 307ms |

Evaluated on 5,000 samples (3,222 attack, 1,778 benign) from the merged dataset.

---

## Prerequisites

- Python 3.11+
- macOS (for menu bar app and identity resolution via Contacts)
- Chrome (for extension)
- ~500 MB RAM for models

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd hybrid_se

# Install dependencies
pip install -r requirements.txt --break-system-packages
pip install -r menubar_app/requirements_menubar.txt --break-system-packages
```

Pre-trained Layer 2 models (`.pkl`) must be placed in the `models/` directory:

```
models/
  tfidf_vectorizer.pkl
  stage1a_svm_final.pkl
  stage1b_lr_final.pkl
```

The NLI model (`cross-encoder/nli-deberta-v3-small`) is downloaded automatically on first run from HuggingFace (~90 MB).

---

## Running the System

### 1. Start the server

```bash
# From project root — must be running before any client
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Wait for: `Pipeline ready in X.Xs`

### 2. Start the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Opens at `http://localhost:8501`. Includes pre-built demo examples for presentations.

### 3. Load the Chrome extension

```
Chrome → chrome://extensions → Developer mode ON
→ Load unpacked → select the extension/ folder
```

Works on Gmail, Outlook Web, and WhatsApp Web. Green dot in popup = server online.

### 4. Start the macOS menu bar app

```bash
python menubar_app/se_menubar.py
```

Scans clipboard text from macOS Messages and WhatsApp Desktop.

---

## macOS Permissions (menu bar app)

Two permissions required in **System Preferences → Privacy & Security**:

| Permission | Purpose |
|---|---|
| Accessibility | Reads contact name from Messages / WhatsApp Desktop UI |
| Contacts | Maps contact name to email for cross-platform identity resolution |

Verify after granting:

```bash
python3 -c "from ApplicationServices import AXIsProcessTrusted; print('AX:', AXIsProcessTrusted())"
python3 -c "from Contacts import CNContactStore; print('Contacts:', CNContactStore.authorizationStatusForEntityType_(0))"
# Both should return True / 3
```

---

## Testing

```bash
# End-to-end tests (requires server running)
python test_e2e.py
# Expected: 31 passed  0 failed

# Layer 4 unit tests (no server needed)
python layer4/test_layer4.py
# Expected: 22 passed  0 failed

# Full pipeline test (loads L2 + L3 models)
python layer4/test_layer4.py --full
```

---

## CLI Usage

```bash
# Single message
python run_pipeline.py --conv-id conv_01 \
  -m "Urgent: verify your account now."

# Multi-message conversation (tests pattern detection)
python run_pipeline.py --conv-id conv_42 \
  -m "Hi I'm from IT helpdesk, detected unusual activity" \
  -m "Just a routine security check, nothing to worry about" \
  -m "Need your login credentials to complete the verification"

# JSON output
python run_pipeline.py --conv-id conv_42 \
  -m "Wire the funds before EOD" \
  --json
```

---

## API Reference

Server runs at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/scan` | POST | Scan one message. Maintains sliding window state. |
| `/scan/thread` | POST | Scan a full conversation at once. |
| `/conversation/{id}` | GET | Get current window assessment. |
| `/conversation/{id}` | DELETE | Reset conversation window. |
| `/conversations` | GET | List all active windows. |
| `/health` | GET | Server and pipeline status. |

**POST /scan request body:**

```json
{
  "text":            "Message text to scan",
  "conversation_id": "conv_42",
  "sender_name":     "John Smith",
  "sender_email":    "john@example.com",
  "platform":        "gmail",
  "message_id":      "msg_001",
  "timestamp":       "2026-05-12T09:15:00Z"
}
```

**Response:**

```json
{
  "conversation_id":  "entity_a3f9b2c1",
  "entity_risk":      67,
  "alert_level":      "HIGH",
  "attack_pattern":   "authority_then_credential",
  "dominant_label":   "pretexting",
  "confidence":       0.41,
  "reasons":          ["m1: pretexting (0.28) ...", "m3: credential_harvesting (0.64) ..."],
  "window_size":      3,
  "last_message":     { "label": "credential_harvesting", "confidence": 0.64, "reason": "..." }
}
```

---

## Project Structure

```
hybrid_se/
├── layer3_slm/
│   ├── config_layer3.py          Attack labels, hypothesis templates, thresholds
│   └── src/
│       ├── nli_classifier.py     Zero-shot NLI batched forward pass
│       ├── explainer.py          Signal extraction + reason generation
│       ├── layer3_pipeline.py    Layer 3 orchestrator
│       └── evaluate_nli.py       Evaluation script
├── layer4/
│   ├── layer4a_risk_counter.py   Conversation risk accumulator
│   ├── layer4b_semantic_window.py Pattern detection + entity risk
│   ├── hybrid_pipeline.py        Full L1→L2→L4a→L3→L4b wiring
│   └── test_layer4.py            22 unit tests
├── server/
│   └── main.py                   FastAPI server, 6 endpoints
├── extension/
│   ├── manifest.json             v1.1.0 — Gmail, Outlook, WhatsApp Web
│   ├── content.js                DOM extraction for all platforms
│   ├── popup.js / popup.html     Scan UI
│   ├── details.html / details.js Full breakdown window
│   └── background.js             Badge colour management
├── menubar_app/
│   ├── se_menubar.py             macOS menu bar app (rumps)
│   ├── accessibility.py          AppleScript + AX API contact detection
│   └── identity.py               CNContactStore cross-platform resolver
├── dashboard/
│   └── streamlit_app.py          Analyst dashboard with demo presets
├── test_e2e.py                   31 end-to-end HTTP tests
├── run_pipeline.py               CLI for single/multi-message processing
└── requirements.txt
```

---

## How the Sliding Window Works

Single-message classifiers miss multi-step SE attacks. The sliding window maintains the last 10 Layer 3 outputs per conversation and detects attack patterns across messages:

```
Message 1: "Hi I'm from IT helpdesk"      → pretexting  (0.28)  LOW
Message 2: "Just a routine security check" → pretexting  (0.30)  LOW
Message 3: "Need your login credentials"   → credential_harvesting (0.64)  HIGH
                                             pattern: authority_then_credential
                                             entity_risk: 67/100
```

Cross-platform: Messages, Gmail, and WhatsApp from the same contact (resolved via macOS Contacts) share one sliding window. A trust-building sequence on iMessage followed by a credential request via Gmail will be detected as a single attack.

---

## Configuration

Key values in `layer3_slm/config_layer3.py`:

```python
LAYER2_THRESHOLD          = 0     # all SVM-flagged messages go to NLI
MIN_SUBTYPE_CONFIDENCE    = 0.25  # below this, falls back to "phishing"
TOP_LABELS_MIN_CONF       = 0.15  # minimum for top_labels inclusion
SUSPICIOUS_THRESHOLD      = 3.0   # L4a: triggers SVM override at this accumulation
```

To add a new attack type — no retraining required:

```python
# In config_layer3.py:
ATTACK_LABELS.append("smishing")
HYPOTHESIS_TEMPLATES["smishing"] = (
    "The sender is impersonating a legitimate service via SMS "
    "and directing the recipient to click a malicious link."
)
```

---

## Layer 3 Model Files

The Layer 3 source code is included in this repository, while its large pretrained
model/checkpoint artifacts are hosted separately to keep the GitHub repository
lightweight.

**Google Drive:** https://drive.google.com/drive/folders/1IqKVwEKu6EduP2JhEY-YCQ9Fg0zVjT9Y

After downloading the Layer 3 files, place them under `layer3_slm/model/`.
The `.gitignore` file intentionally excludes this directory and large model artifacts.

## Known Limitations

- **Non-English phishing**: SVM trained on English corpus. Indian English or non-standard phrasing may be called benign by Layer 2.
- **Single-message confidence**: NLI confidence is often 0.25–0.45 on isolated messages by design — the system is optimised for multi-message conversations.
- **macOS only**: Menu bar app and cross-platform identity resolution require macOS Contacts and Accessibility frameworks.

---

## License

Academic project — Mini Project, Semester 6.
