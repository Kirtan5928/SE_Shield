# SE Shield — Hybrid Social Engineering Detection System

> A multi-layer, context-aware system for detecting social engineering attacks across email and messaging platforms using classical machine learning, zero-shot natural language inference, and conversation-level behavioral analysis.

SE Shield is a hybrid social engineering detection pipeline designed to identify both **single-message threats** and **multi-step social engineering campaigns** that become apparent only when multiple messages are analyzed together.

The system combines traditional machine learning with semantic NLI classification and a sliding-window context engine, while providing multiple interfaces for real-world use.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Detection Pipeline](#detection-pipeline)
* [Supported Attack Types](#supported-attack-types)
* [Evaluation Results](#evaluation-results)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Model Files](#model-files)
* [Running the System](#running-the-system)
* [Chrome Extension](#chrome-extension)
* [macOS Menu Bar Application](#macos-menu-bar-application)
* [Dashboard](#dashboard)
* [CLI Usage](#cli-usage)
* [API Reference](#api-reference)
* [Testing](#testing)
* [Conversation-Level Detection](#conversation-level-detection)
* [Configuration](#configuration)
* [Security and Privacy](#security-and-privacy)
* [Known Limitations](#known-limitations)
* [Future Improvements](#future-improvements)
* [License](#license)

---

## Overview

Traditional phishing detection systems often evaluate messages independently. This approach can fail when an attacker deliberately spreads a social engineering attack across multiple messages.

For example:

```text
Message 1
"Hi, I'm from the IT helpdesk. We've detected unusual activity."

        ↓

Message 2
"Don't worry, this is just a routine security verification."

        ↓

Message 3
"Please provide your login credentials so I can complete the verification."
```

Individually, the first two messages may appear relatively harmless.

Together, however, they form a clear social engineering sequence:

**Authority → Trust Building → Credential Request**

SE Shield addresses this problem by maintaining conversation context and combining message-level predictions with semantic and behavioral signals.

---

# Key Features

### Multi-Layer Detection

The system combines multiple detection mechanisms:

* Unicode normalization and homoglyph handling
* TF-IDF based text representation
* SVM-based binary classification
* Logistic Regression risk scoring
* Zero-shot NLI classification
* Conversation-level risk accumulation
* Sliding-window semantic analysis
* Attack-pattern recognition
* Cross-platform identity resolution

### Zero-Shot Attack Classification

Layer 3 uses a Natural Language Inference model to classify messages into social engineering categories without requiring a dedicated labelled training dataset for every attack type.

### Conversation-Level Detection

The system maintains recent messages within a conversation and analyzes their combined behavior.

This allows it to identify attacks that develop gradually rather than relying only on individual-message classification.

### Cross-Platform Detection

The architecture supports analysis across:

* Gmail
* Outlook Web
* WhatsApp Web
* WhatsApp Desktop
* macOS Messages

Cross-platform identity resolution allows messages associated with the same contact to contribute to a shared conversation context.

### Multiple Interfaces

SE Shield provides:

* FastAPI backend
* Streamlit analyst dashboard
* Chrome browser extension
* macOS menu bar application
* Command-line interface

### Local Inference

The detection pipeline is designed around local model inference rather than requiring an external AI API for every message.

---

# System Architecture

```text
                         Incoming Message
                                │
                                ▼
                    ┌──────────────────────┐
                    │       Layer 1        │
                    │    Preprocessing     │
                    │                      │
                    │ Unicode Normalisation│
                    │ Homoglyph Handling   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Layer 2        │
                    │      ML Triage       │
                    │                      │
                    │ TF-IDF               │
                    │ SVM Binary Gate     │
                    │ LR Risk Score       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Layer 4a        │
                    │    Risk Counter      │
                    │                      │
                    │ Conversation Risk    │
                    │ Accumulation         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Layer 3        │
                    │      NLI Engine      │
                    │                      │
                    │ Zero-Shot Semantic   │
                    │ Attack Classification│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Layer 4b        │
                    │  Semantic Window     │
                    │                      │
                    │ Pattern Detection    │
                    │ Entity Risk          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Interfaces      │
                    │                      │
                    │ FastAPI              │
                    │ Dashboard            │
                    │ Chrome Extension     │
                    │ macOS Menu Bar       │
                    └──────────────────────┘
```

---

# Detection Pipeline

## Layer 1 — Preprocessing

The first layer normalizes incoming text before classification.

Responsibilities include:

* Unicode normalization
* Homoglyph substitution
* Text normalization
* Preparation of input for downstream classifiers

This reduces the ability of attackers to evade detection through visually similar Unicode characters or formatting variations.

---

## Layer 2 — Machine Learning Triage

Layer 2 provides the initial machine-learning based assessment.

The layer uses:

* TF-IDF feature extraction
* SVM binary classification
* Logistic Regression risk scoring

Its purpose is to provide an efficient first-stage assessment and route suspicious messages into deeper semantic analysis.

---

## Layer 3 — Zero-Shot NLI Engine

Layer 3 performs semantic attack classification using a Natural Language Inference model.

The current implementation uses:

```text
cross-encoder/nli-deberta-v3-small
```

The classifier evaluates the relationship between the message and predefined attack hypotheses.

This allows the system to classify attack sub-types without requiring a separate supervised classifier for every category.

---

## Layer 4a — Conversation Risk Counter

Layer 4a maintains accumulated risk across messages within a conversation.

This is important because a sequence of individually low-risk messages can become high-risk when considered together.

The risk counter also provides an additional signal that can override an isolated low-confidence classification when the conversation has accumulated sufficient evidence.

---

## Layer 4b — Semantic Window

Layer 4b analyzes recent Layer 3 outputs within a sliding conversation window.

It identifies higher-level behavioral patterns such as:

```text
trust_build_then_exploit
authority_then_credential
bec_sequence
urgency_escalation
delayed_execution
multi_vector
```

The resulting analysis produces an entity-level risk score and an overall alert level.

---

# Supported Attack Types

Layer 3 currently evaluates the following categories:

| Attack Type               | Description                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| Phishing                  | Fraudulent messages attempting to deceive the recipient                                    |
| Spear Phishing            | Targeted phishing directed at a specific individual                                        |
| Pretexting                | Social engineering based on a fabricated identity or scenario                              |
| Credential Harvesting     | Attempts to obtain passwords, credentials, or authentication information                   |
| Baiting                   | Malicious offers or incentives designed to induce user interaction                         |
| Vishing                   | Voice-oriented social engineering or impersonation scenarios                               |
| Business Email Compromise | Fraudulent requests involving organizational authority, payments, or sensitive information |

---

# Evaluation Results

The integrated system was evaluated on **5,000 samples**:

| Metric        |     Result |
| ------------- | ---------: |
| Attack Recall | **0.9969** |
| Precision     | **0.9932** |
| F1 Score      | **0.9950** |
| Accuracy      | **0.9936** |
| Latency — p50 | **143 ms** |
| Latency — p95 | **307 ms** |

Dataset composition:

```text
Total Samples : 5,000
Attack        : 3,222
Benign        : 1,778
```

The results demonstrate that the integrated architecture can achieve high detection performance while maintaining relatively low inference latency.

---

# Technology Stack

| Component               | Technology                         |
| ----------------------- | ---------------------------------- |
| Programming Language    | Python                             |
| Machine Learning        | Scikit-learn                       |
| Semantic Classification | PyTorch / Transformers             |
| NLI Model               | DeBERTa NLI                        |
| Backend                 | FastAPI                            |
| Dashboard               | Streamlit                          |
| Browser Extension       | JavaScript / Chrome Extension APIs |
| macOS Integration       | Python / Accessibility APIs        |
| Identity Resolution     | macOS Contacts Framework           |
| Version Control         | Git / GitHub                       |

---

# Project Structure

```text
SE_Shield/
│
├── app/
│
├── dashboard/
│   └── streamlit_app.py
│
├── extension/
│   ├── manifest.json
│   ├── content.js
│   ├── popup.html
│   ├── popup.js
│   ├── details.html
│   ├── details.js
│   └── background.js
│
├── layer3_slm/
│   ├── config_layer3.py
│   ├── README.md
│   └── src/
│       ├── nli_classifier.py
│       ├── explainer.py
│       ├── layer3_pipeline.py
│       └── evaluate_nli.py
│
├── layer4/
│   ├── layer4a_risk_counter.py
│   ├── layer4b_semantic_window.py
│   ├── hybrid_pipeline.py
│   └── test_layer4.py
│
├── menubar_app/
│   ├── se_menubar.py
│   ├── accessibility.py
│   └── identity.py
│
├── server/
│   └── main.py
│
├── src/
│
├── config.py
├── requirements.txt
├── run_pipeline.py
├── run_layer4.py
├── test_e2e.py
├── .gitignore
└── README.md
```

---

# Installation

## Prerequisites

Recommended environment:

* Python 3.11+
* Git
* Google Chrome
* macOS for the menu bar application and contact-based identity resolution

Clone the repository:

```bash
git clone https://github.com/Kirtan5928/SE_Shield.git
cd SE_Shield
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

For the macOS menu bar application:

```bash
pip install -r menubar_app/requirements_menubar.txt
```

---

# Model Files

Large model artifacts are intentionally **not stored in this GitHub repository**.

This keeps the repository lightweight and avoids committing large binary files to Git.

## Layer 3

The Layer 3 model/checkpoint files are hosted separately on Google Drive.

### Download Layer 3 Models

**Google Drive:**

https://drive.google.com/drive/folders/1IqKVwEKu6EduP2JhEY-YCQ9Fg0zVjT9Y

After downloading the required files, place them under:

```text
layer3_slm/
└── model/
    └── <Layer 3 model files>
```

The repository's `.gitignore` intentionally excludes the model directory and large ML artifacts.

## Layer 2

Layer 2 model artifacts are also excluded from GitHub.

The expected model directory is:

```text
models/
├── tfidf_vectorizer.pkl
├── stage1a_svm_final.pkl
└── stage1b_lr_final.pkl
```

Obtain these files from the project maintainer and place them in the `models/` directory.

---

# Running the System

## 1. Start the FastAPI Server

From the project root:

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

The server should become available at:

```text
http://127.0.0.1:8000
```

---

## 2. Start the Dashboard

In another terminal:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

The dashboard provides an analyst-oriented interface for testing messages and demonstrating the detection pipeline.

---

# Chrome Extension

The browser extension can be loaded as an unpacked Chrome extension.

### Installation

1. Open Chrome.
2. Navigate to:

```text
chrome://extensions
```

3. Enable **Developer mode**.
4. Select **Load unpacked**.
5. Select:

```text
SE_Shield/extension/
```

The extension supports web-based communication environments implemented by the project, including:

* Gmail
* Outlook Web
* WhatsApp Web

The extension communicates with the locally running SE Shield backend.

---

# macOS Menu Bar Application

The macOS menu bar application provides integration with supported desktop messaging applications.

Run:

```bash
python menubar_app/se_menubar.py
```

The application can process clipboard/message content and resolve contact identity using macOS system frameworks.

## Required macOS Permissions

The following permissions may be required under:

**System Settings → Privacy & Security**

| Permission    | Purpose                              |
| ------------- | ------------------------------------ |
| Accessibility | Allows UI inspection and interaction |
| Contacts      | Enables contact identity resolution  |

The application uses these permissions to associate messages with known contacts and improve cross-platform conversation tracking.

---

# Dashboard

The Streamlit dashboard provides an interactive interface for:

* Message scanning
* Risk visualization
* Attack-type classification
* Conversation analysis
* Demonstration scenarios
* Pipeline output inspection

Start it with:

```bash
streamlit run dashboard/streamlit_app.py
```

---

# CLI Usage

SE Shield can also be used directly from the command line.

## Scan a Single Message

```bash
python run_pipeline.py \
  --conv-id conv_01 \
  -m "Urgent: verify your account now."
```

## Scan a Multi-Message Conversation

```bash
python run_pipeline.py \
  --conv-id conv_42 \
  -m "Hi I'm from IT helpdesk, detected unusual activity" \
  -m "Just a routine security check, nothing to worry about" \
  -m "Need your login credentials to complete the verification"
```

This example demonstrates the type of multi-message sequence that the conversation-level engine is designed to detect.

## JSON Output

```bash
python run_pipeline.py \
  --conv-id conv_42 \
  -m "Wire the funds before EOD" \
  --json
```

---

# API Reference

The FastAPI server runs at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Endpoint             | Method | Purpose                          |
| -------------------- | ------ | -------------------------------- |
| `/scan`              | POST   | Scan an individual message       |
| `/scan/thread`       | POST   | Scan a complete conversation     |
| `/conversation/{id}` | GET    | Retrieve conversation risk state |
| `/conversation/{id}` | DELETE | Reset conversation state         |
| `/conversations`     | GET    | List active conversations        |
| `/health`            | GET    | Check system health              |

---

## Example `/scan` Request

```json
{
  "text": "Urgent: verify your account immediately.",
  "conversation_id": "conv_42",
  "sender_name": "John Smith",
  "sender_email": "john@example.com",
  "platform": "gmail",
  "message_id": "msg_001",
  "timestamp": "2026-05-12T09:15:00Z"
}
```

## Example Response

```json
{
  "conversation_id": "entity_a3f9b2c1",
  "entity_risk": 67,
  "alert_level": "HIGH",
  "attack_pattern": "authority_then_credential",
  "dominant_label": "pretexting",
  "confidence": 0.41,
  "window_size": 3
}
```

---

# Conversation-Level Detection

One of the primary design goals of SE Shield is detecting attacks that unfold across multiple messages.

Consider:

```text
┌────────────────────────────────────────────────────┐
│ Message 1                                           │
│ "I'm from the IT helpdesk."                        │
│                                                     │
│ → Pretexting                                       │
│ → Low immediate risk                               │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ Message 2                                           │
│ "We've detected unusual activity on your account."│
│                                                     │
│ → Authority / urgency                               │
│ → Risk increases                                    │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ Message 3                                           │
│ "Send your login credentials for verification."    │
│                                                     │
│ → Credential harvesting                             │
│ → High risk                                         │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
              Pattern Detected
         authority_then_credential

                 Entity Risk: 67/100
                    Alert: HIGH
```

The sliding-window mechanism allows the system to consider the **sequence of actions**, rather than only the content of an individual message.

---

# Attack Pattern Detection

The semantic window currently supports patterns including:

### `trust_build_then_exploit`

An attacker first establishes credibility before attempting exploitation.

### `authority_then_credential`

An attacker establishes authority or legitimacy and subsequently requests authentication information.

### `bec_sequence`

A sequence resembling business email compromise behavior.

### `urgency_escalation`

The attacker progressively increases pressure and urgency.

### `delayed_execution`

The attacker establishes context before initiating a malicious action at a later stage.

### `multi_vector`

Multiple social engineering signals appear across different attack vectors.

---

# Configuration

Layer 3 configuration is primarily controlled through:

```text
layer3_slm/config_layer3.py
```

Important parameters include:

```python
LAYER2_THRESHOLD = 0
MIN_SUBTYPE_CONFIDENCE = 0.25
TOP_LABELS_MIN_CONF = 0.15
SUSPICIOUS_THRESHOLD = 3.0
```

These values influence:

* Layer 2 routing
* Minimum semantic classification confidence
* Top-label inclusion
* Conversation-level risk escalation

Attack categories and their corresponding semantic hypotheses can also be extended through the Layer 3 configuration.

---

# Security and Privacy

SE Shield is designed with local processing in mind.

### Local Processing

The core detection pipeline is designed to run locally without requiring every message to be sent to a third-party AI inference API.

### Model Isolation

Large model artifacts are kept outside the Git repository to reduce repository size and prevent accidental distribution of model binaries.

### Credential Protection

Secrets, API keys, environment files, credentials, and private key material should never be committed to the repository.

The `.gitignore` file contains exclusions for common secret and environment-file formats.

### Browser Integration

The browser extension communicates with the local SE Shield backend rather than directly exposing model infrastructure to web pages.

---

# Testing

## End-to-End Tests

With the FastAPI server running:

```bash
python test_e2e.py
```

## Layer 4 Unit Tests

```bash
python layer4/test_layer4.py
```

## Full Pipeline Tests

Tests requiring the Layer 2 and Layer 3 models:

```bash
python layer4/test_layer4.py --full
```

---

# Known Limitations

### English-Centric Training

The current Layer 2 model is primarily trained on English-language data.

Messages written in other languages or heavily non-standard forms of English may not receive optimal classification.

### Single-Message Confidence

Some social engineering messages are intentionally ambiguous when viewed individually.

The system is therefore designed to benefit from conversation-level context rather than relying solely on a single-message confidence score.

### macOS Dependency

The menu bar application and cross-platform identity resolution rely on macOS-specific frameworks.

The core backend and detection pipeline are not inherently limited to macOS, but these integrations are.

### Model Availability

The complete pipeline requires the relevant pretrained model artifacts. These are intentionally hosted separately from GitHub.

---

# Future Improvements

Potential future extensions include:

* Multilingual social engineering detection
* Additional messaging-platform integrations
* Improved cross-platform identity correlation
* More sophisticated temporal analysis
* Online learning and model adaptation
* Additional social engineering attack categories
* Explainable AI improvements
* Centralized enterprise deployment
* Containerized deployment
* Automated model/version management
* More extensive adversarial robustness testing

---

# Project Status

SE Shield is an academic engineering project demonstrating a hybrid approach to social engineering detection.

The current implementation includes:

* Multi-layer detection pipeline
* Machine-learning triage
* Zero-shot NLI classification
* Conversation-level risk accumulation
* Semantic attack-pattern detection
* FastAPI backend
* Streamlit dashboard
* Chrome extension
* macOS menu bar application
* Cross-platform identity resolution

---

# Repository

GitHub:

https://github.com/Kirtan5928/SE_Shield

Layer 3 model files:

https://drive.google.com/drive/folders/1IqKVwEKu6EduP2JhEY-YCQ9Fg0zVjT9Y

---

# License

Academic project — Mini Project, Semester 6.

The repository is intended primarily for academic evaluation, demonstration, and research purposes.
