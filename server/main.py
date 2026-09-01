"""
server/main.py
===============
FastAPI local server — wraps HybridPipeline in 3 REST endpoints.

Endpoints
---------
POST /scan                → process one message, returns L4b assessment
GET  /conversation/{id}   → get current window assessment without adding a message
DELETE /conversation/{id} → reset window (thread resolved / false alarm dismissed)
GET  /health              → server status + active conversation count

Run
---
  # From project root (hybrid_se/):
  uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload

  # Production (no reload):
  uvicorn server.main:app --host 127.0.0.1 --port 8000 --workers 1

  # The server binds to localhost only — not accessible from outside the machine.
  # This is intentional: all inference runs locally, no data leaves the machine.

CORS
----
Allows requests from:
  - Chrome extensions (chrome-extension://)
  - Localhost (for Streamlit dashboard)
  - File:// (for local HTML test pages)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup ─────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # server/
_PROJECT_ROOT = _HERE.parent                      # hybrid_se/
_LAYER3_DIR   = _PROJECT_ROOT / "layer3_slm"
_LAYER4_DIR   = _PROJECT_ROOT / "layer4"

for p in [str(_LAYER3_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Force layer3_slm at position 0
_l3_str = str(_LAYER3_DIR)
if _l3_str in sys.path:
    sys.path.remove(_l3_str)
sys.path.insert(0, _l3_str)

from layer4.hybrid_pipeline import HybridPipeline

# Identity resolver — maps contact name / email → canonical entity_id
# Enables cross-platform conversation windows (Messages + Gmail + WA → same window)
# Gracefully falls back if Contacts permission not granted or pyobjc not installed
try:
    # Do NOT sys.path.insert here — project root is already in sys.path
    # when running via uvicorn from project root. Inserting it at [0]
    # would push layer3_slm down and break src.layer3_pipeline imports.
    from menubar_app.identity import resolve as _resolve_identity
    _IDENTITY_AVAILABLE = True
except ImportError:
    _IDENTITY_AVAILABLE = False
    def _resolve_identity(name=None, email=None, phone=None):
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("se_server")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SE Detection Server",
    description="Hybrid social engineering detection pipeline — local inference only.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dashboard
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # Dev HTML
        "http://127.0.0.1:3000",
        "*",                       # Chrome extensions (chrome-extension://)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline singleton (loaded once at startup) ────────────────────────────
_pipeline: Optional[HybridPipeline] = None
_startup_time: float = 0.0


@app.on_event("startup")
async def load_pipeline() -> None:
    global _pipeline, _startup_time
    t0 = time.perf_counter()
    logger.info("Loading HybridPipeline …")
    _pipeline = HybridPipeline()
    _startup_time = time.perf_counter() - t0
    logger.info("Pipeline ready in %.1fs", _startup_time)


def get_pipeline() -> HybridPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")
    return _pipeline


# ── Request / Response models ──────────────────────────────────────────────

class ScanRequest(BaseModel):
    text:            str            = Field(..., description="Raw message text to scan")
    conversation_id: str            = Field(..., description="Thread/session ID. Phase 1: any string. Phase 2: resolved to entity_id via identity.py")
    message_id:      Optional[str]  = Field(None, description="Unique message ID (auto-assigned if None)")
    timestamp:       Optional[str]  = Field(None, description="ISO-8601 timestamp")
    # Phase 2 cross-platform identity fields — forwarded by Chrome extension
    # and menubar app. Server will resolve these to a canonical entity_id
    # via macOS CNContactStore when identity.py is implemented.
    sender_name:     Optional[str]  = Field(None, description="Contact display name (from WA Web header, Messages AX, etc.)")
    sender_email:    Optional[str]  = Field(None, description="Sender email address (from Gmail sender field)")
    platform:        Optional[str]  = Field("unknown", description="gmail|whatsapp_web|outlook_live|outlook_office|imessage|manual")

    class Config:
        json_schema_extra = {
            "example": {
                "text":            "Urgent: your account will be suspended. Verify now.",
                "conversation_id": "gmail_thread_18e4f2a3b1c9d7e0",
                "message_id":      "msg_001",
                "timestamp":       "2026-05-11T09:15:00Z",
                "sender_name":     "John Smith",
                "sender_email":    "john.smith@company.com",
                "platform":        "gmail",
            }
        }


class MessageResult(BaseModel):
    message_id:      Optional[str]
    timestamp:       Optional[str]
    label:           str
    confidence:      float
    top_labels:      list
    reason:          str
    layer2_risk:     int
    latency_ms:      float


class ConversationAssessment(BaseModel):
    conversation_id: str
    entity_risk:     int
    attack_pattern:  str
    alert_level:     str
    dominant_label:  str
    confidence:      float
    reasons:         list[str]
    window_size:     int
    message_ids:     list
    last_message:    Optional[dict]  = None
    l4a_context:     Optional[dict]  = None


class HealthResponse(BaseModel):
    status:              str
    pipeline_ready:      bool
    active_conversations: int
    startup_time_s:      float


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — confirms server and pipeline are running."""
    pipe = _pipeline
    active = len(pipe._l4b._windows) if pipe else 0
    return HealthResponse(
        status="ok",
        pipeline_ready=pipe is not None,
        active_conversations=active,
        startup_time_s=round(_startup_time, 2),
    )


@app.post("/scan", response_model=ConversationAssessment)
async def scan(req: ScanRequest) -> ConversationAssessment:
    """
    Scan one message and return updated conversation assessment.

    This is the primary endpoint — called by the Chrome extension on each
    email scan and by the Streamlit dashboard on each message submission.

    The sliding window state is maintained server-side, keyed by conversation_id.
    Repeated calls with the same conversation_id accumulate context.
    """
    pipe = get_pipeline()

    # ── Cross-platform identity resolution ────────────────────────────────
    # If sender_name or sender_email is provided, attempt to resolve to a
    # canonical entity_id via macOS Contacts (CNContactStore).
    # This unifies Messages, Gmail, and WhatsApp Web into a single sliding
    # window when they belong to the same real-world contact.
    #
    # Resolution priority:
    #   1. sender_email provided → use directly as canonical (already unique)
    #   2. sender_name provided  → CNContactStore lookup → email → entity_id
    #   3. No identity fields    → use raw conversation_id (platform-isolated)
    #   4. identity.py not installed (no pyobjc) → use raw conversation_id
    conv_id = req.conversation_id  # default: client-provided platform-specific ID

    if _IDENTITY_AVAILABLE and (req.sender_name or req.sender_email):
        resolved = _resolve_identity(
            name  = req.sender_name,
            email = req.sender_email,
        )
        if resolved:
            conv_id = resolved
            logger.info(
                "Identity resolved: sender='%s' / '%s' → conv_id='%s'",
                req.sender_name or "", req.sender_email or "", conv_id,
            )

    # ── Auto-assign message_id ────────────────────────────────────────────
    msg_id = req.message_id
    if not msg_id:
        win = pipe._l4b._windows.get(conv_id)
        n   = len(win) + 1 if win else 1
        msg_id = f"{conv_id}_msg_{n:03d}"

    logger.info(
        "scan  platform=%s  conv=%s  msg=%s  sender=%s  text='%s…'",
        req.platform or "unknown",
        conv_id,
        msg_id,
        req.sender_name or req.sender_email or "unknown",
        req.text[:60],
    )

    result = pipe.process(
        text            = req.text,
        conversation_id = conv_id,
        message_id      = msg_id,
        timestamp       = req.timestamp,
    )

    return ConversationAssessment(**result)


@app.get("/conversation/{conversation_id}", response_model=ConversationAssessment)
async def get_conversation(conversation_id: str) -> ConversationAssessment:
    """
    Get current assessment for a conversation without adding a new message.
    Returns 404 if conversation does not exist or was reset.
    """
    pipe       = get_pipeline()
    assessment = pipe._l4b.assess(conversation_id)
    l4a_ctx    = pipe._l4a.get_context(conversation_id)

    # window_size == 0 means conversation does not exist or was reset
    if assessment.get("window_size", 0) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found or has been reset.",
        )

    result = {**assessment, "last_message": None, "l4a_context": l4a_ctx}
    return ConversationAssessment(**result)


@app.delete("/conversation/{conversation_id}")
async def reset_conversation(conversation_id: str) -> dict:
    """
    Reset the sliding window for a conversation.
    Call this when a thread is resolved, dismissed, or a new analysis is needed.
    """
    pipe = get_pipeline()
    pipe.reset_conversation(conversation_id)
    logger.info("Reset conversation: %s", conversation_id)
    return {"status": "reset", "conversation_id": conversation_id}


@app.get("/conversations")
async def list_conversations() -> dict:
    """List all active conversation windows (for dashboard overview panel)."""
    pipe = get_pipeline()
    return {
        "active": pipe.active_conversations(),
        "count":  len(pipe._l4b._windows),
    }