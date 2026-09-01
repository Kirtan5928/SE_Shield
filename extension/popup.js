/**
 * extension/popup.js
 * Popup logic — sends messages to content.js, calls server, updates UI
 */
"use strict";

const SERVER = "http://127.0.0.1:8000";

// ── DOM refs ──────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const serverDot    = $("serverDot");
const alertBanner  = $("alertBanner");
const alertLevel   = $("alertLevel");
const alertPattern = $("alertPattern");
const riskRow      = $("riskRow");
const riskFill     = $("riskFill");
const riskValue    = $("riskValue");
const statusMsg    = $("statusMsg");
const scanBtn      = $("scanBtn");
const resetBtn     = $("resetBtn");
const detailsBtn   = $("detailsBtn");
const windowInfo   = $("windowInfo");
const windowCount  = $("windowCount");
const convIdDisplay = $("convIdDisplay");
const msgDetail    = $("msgDetail");
const detailLabel  = $("detailLabel");
const detailConf   = $("detailConf");
const detailReason = $("detailReason");
const reasonsSection = $("reasonsSection");
const reasonsList  = $("reasonsList");

let currentConvId  = null;

// ── Server health check ───────────────────────────────────────────────────
async function checkServer() {
  try {
    const r = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      serverDot.className = "server-dot online";
      setStatus("Ready to scan.");
      scanBtn.disabled = false;
      return true;
    }
  } catch (_) {}
  serverDot.className = "server-dot offline";
  setStatus("Server offline. Run: uvicorn server.main:app --host 127.0.0.1 --port 8000", "error");
  return false;
}

// ── Status helpers ────────────────────────────────────────────────────────
function setStatus(msg, type = "") {
  statusMsg.textContent = msg;
  statusMsg.className   = `status ${type}`;
  statusMsg.style.display = "block";
}
function hideStatus() { statusMsg.style.display = "none"; }

// ── UI update from assessment ─────────────────────────────────────────────
function updateUI(result) {
  const level   = result.alert_level   || "LOW";
  const risk    = result.entity_risk   || 0;
  const pattern = result.attack_pattern || "none";
  const last    = result.last_message  || {};

  // Alert banner
  alertBanner.className = `alert-banner visible ${level}`;
  alertLevel.className  = `alert-level ${level}`;
  alertLevel.textContent  = level;
  alertPattern.textContent = pattern.replace(/_/g, " ").toUpperCase();

  // Risk bar
  riskRow.className = "risk-row visible";
  const riskColour = risk >= 80 ? "#E05252" : risk >= 60 ? "#E05252"
                   : risk >= 35 ? "#F0943A" : "#3DC97A";
  riskFill.style.width      = `${risk}%`;
  riskFill.style.background = riskColour;
  riskValue.style.color     = riskColour;
  riskValue.textContent     = `${risk}`;

  // Per-message detail
  if (last.label) {
    msgDetail.className    = "msg-detail visible";
    detailLabel.textContent = last.label.toUpperCase().replace(/_/g, " ");
    detailConf.textContent  = `${(last.confidence * 100).toFixed(0)}%`;
    detailReason.textContent = (last.reason || "").slice(0, 120);
  }

  // Reasoning trail
  const reasons = result.reasons || [];
  if (reasons.length > 0) {
    reasonsSection.className = "reasons-section visible";
    reasonsList.innerHTML = reasons.map(r => {
      const isBenign = r.includes("benign");
      return `<div class="reason-item ${isBenign ? 'benign' : 'attack'}">${r}</div>`;
    }).join("");
  }

  // Window info
  windowInfo.style.display = "flex";
  windowCount.textContent  = `${result.window_size || 0} message${result.window_size !== 1 ? "s" : ""} in window`;
  if (currentConvId) {
    convIdDisplay.textContent = currentConvId.slice(0, 20) + "…";
  }

  resetBtn.style.display = "block";

  // Update badge via background
  chrome.runtime.sendMessage({ type: "UPDATE_BADGE", alert_level: level });

  hideStatus();
}

// ── Scan flow ─────────────────────────────────────────────────────────────
async function runScan() {
  scanBtn.disabled    = true;
  scanBtn.className   = "scan-btn scanning";
  scanBtn.innerHTML   = '<span class="spinner"></span>SCANNING…';
  setStatus("Extracting email…", "loading");

  // 1. Get active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    setStatus("No active tab found.", "error");
    resetScanBtn(); return;
  }

  // 2. Ask content script to extract email
  // If sendMessage fails, the content script hasn't loaded on this tab yet
  // (tab was open before extension was installed/reloaded).
  // Fallback: inject it programmatically, then retry.
  let extracted;
  try {
    extracted = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_EMAIL" });
  } catch (_) {
    const url = tab.url || "";
    const isEmail = url.includes("mail.google.com") || url.includes("outlook");
    if (!isEmail) {
      setStatus("Open a Gmail or Outlook email first.", "error");
      resetScanBtn(); return;
    }
    // Inject content script and retry once
    setStatus("Injecting scanner…", "loading");
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files:  ["content.js"],
      });
      await new Promise(r => setTimeout(r, 300));  // let script initialise
      extracted = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_EMAIL" });
    } catch (e2) {
      setStatus("Could not inject scanner. Try reloading the Gmail tab (Cmd+R).", "error");
      resetScanBtn(); return;
    }
  }

  if (!extracted.success) {
    setStatus(extracted.error || "Could not extract email.", "error");
    resetScanBtn(); return;
  }

  // Destructure all fields including Phase 2 identity fields
  const {
    text, conversation_id, timestamp, client,
    sender_name, sender_email,
  } = extracted.data;
  currentConvId = conversation_id;

  const platformLabel = {
    gmail:         "Gmail",
    outlook_live:  "Outlook",
    outlook_office:"Outlook",
    whatsapp_web:  "WhatsApp Web",
  }[client] || client;
  setStatus(`Scanning on ${platformLabel}…`, "loading");

  // 3. Send to local server
  // sender_name + sender_email forwarded for Phase 2 cross-platform resolution
  const body = {
    text,
    conversation_id,
    timestamp,
    message_id:   null,
    sender_name:  sender_name  || null,
    sender_email: sender_email || null,
    platform:     client,
  };

  let result;
  try {
    const r = await fetch(`${SERVER}/scan`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(30000),  // 30s — NLI inference can take ~500ms
    });
    if (!r.ok) {
      const err = await r.text();
      setStatus(`Server error: ${err.slice(0, 80)}`, "error");
      resetScanBtn(); return;
    }
    result = await r.json();
  } catch (e) {
    setStatus(`Connection failed: ${e.message}`, "error");
    resetScanBtn(); return;
  }

  // 4. Update UI
  updateUI(result);
  resetScanBtn(client);

  // 5. Store full result for details window
  chrome.storage.session.set({ lastResult: result, lastConvId: conversation_id });
  detailsBtn.style.display = "block";
}

// Platform-aware button label
const SCAN_LABELS = {
  gmail:          "SCAN THIS EMAIL",
  outlook_live:   "SCAN THIS EMAIL",
  outlook_office: "SCAN THIS EMAIL",
  whatsapp_web:   "SCAN THIS MESSAGE",
};

function resetScanBtn(platform) {
  scanBtn.disabled    = false;
  scanBtn.className   = "scan-btn";
  scanBtn.textContent = SCAN_LABELS[platform] || "SCAN THIS MESSAGE";
}

// ── Reset conversation ────────────────────────────────────────────────────
async function resetConversation() {
  if (!currentConvId) return;
  try {
    await fetch(`${SERVER}/conversation/${encodeURIComponent(currentConvId)}`, {
      method: "DELETE",
    });
  } catch (_) {}

  // Clear UI
  alertBanner.className    = "alert-banner";
  riskRow.className        = "risk-row";
  msgDetail.className      = "msg-detail";
  reasonsSection.className = "reasons-section";
  windowInfo.style.display = "none";
  resetBtn.style.display   = "none";
  currentConvId            = null;
  detailsBtn.style.display = "none";
  chrome.storage.session.remove(["lastResult", "lastConvId"]);
  chrome.runtime.sendMessage({ type: "UPDATE_BADGE", alert_level: "LOW" });
  setStatus("Conversation reset. Ready to scan.");
}

// ── Open details window ───────────────────────────────────────────────────
function openDetails() {
  const url = chrome.runtime.getURL("details.html");
  chrome.windows.create({
    url,
    type:   "popup",
    width:  600,
    height: 700,
    focused: true,
  });
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Set platform-aware button label before any scan runs
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = (tab && tab.url) || "";
  if (url.includes("web.whatsapp.com")) {
    resetScanBtn("whatsapp_web");
  } else if (url.includes("mail.google.com") || url.includes("outlook")) {
    resetScanBtn("gmail");
  }
  // else stays "SCAN THIS MESSAGE" — correct for unknown pages

  await checkServer();
  scanBtn.addEventListener("click",     runScan);
  resetBtn.addEventListener("click",    resetConversation);
  detailsBtn.addEventListener("click",  openDetails);

  // Restore details btn if a result is already stored
  chrome.storage.session.get(["lastResult"], (data) => {
    if (data.lastResult) {
      updateUI(data.lastResult);
      currentConvId = data.lastResult.conversation_id;
      detailsBtn.style.display = "block";
    }
  });
});