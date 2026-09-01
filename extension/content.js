/**
 * extension/content.js
 * v1.1.0 — Gmail, Outlook, WhatsApp Web
 *
 * Content script — reads message text and conversation identity from DOM.
 * Injected automatically on supported pages. Falls back to programmatic
 * injection via chrome.scripting.executeScript if tab pre-dates extension.
 *
 * Phase 2 note: sender_name / sender_email are forwarded to the server,
 * which resolves them via macOS Contacts (CNContactStore) into a
 * cross-platform entity_id. For now, conv_id is platform-specific and
 * accumulates correctly per-contact within each platform.
 */
"use strict";

// ── Client detection ────────────────────────────────────────────────────────
function detectClient() {
  const host = window.location.hostname;
  if (host.includes("mail.google.com"))    return "gmail";
  if (host.includes("outlook.live.com"))   return "outlook_live";
  if (host.includes("outlook.office.com")) return "outlook_office";
  if (host.includes("web.whatsapp.com"))   return "whatsapp_web";
  return "unknown";
}

// ────────────────────────────────────────────────────────────────────────────
// GMAIL
// ────────────────────────────────────────────────────────────────────────────
/** djb2 hash — same scheme as WhatsApp conv_id. Safe for any unicode. */
function djb2Hash(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h) ^ str.charCodeAt(i);
    h = h >>> 0;  // keep unsigned 32-bit
  }
  return h.toString(16).padStart(8, "0");
}

function gmailThreadId() {
  // Gmail thread IDs are mixed-case alphanumeric (e.g. FMfcgzQgMMJT…)
  const match = window.location.hash.match(/#[a-z]+\/([A-Za-z0-9_-]+)$/);
  if (match) return `gmail_${match[1].slice(0, 24)}`;
  // Fallback: hash subject — btoa REMOVED, throws on non-Latin1 subjects
  const subject = document.querySelector("h2.hP");
  if (subject) return `gmail_subj_${djb2Hash(subject.innerText)}`;
  return `gmail_${Date.now()}`;
}

function gmailEmailText() {
  const messages = document.querySelectorAll(".a3s.aiL");
  if (!messages.length) {
    const body = document.querySelector(".ii.gt");
    return body ? body.innerText.trim() : "";
  }
  return messages[messages.length - 1].innerText.trim();
}

function gmailSelectedText() {
  const sel = window.getSelection();
  return sel && sel.toString().trim().length > 10 ? sel.toString().trim() : null;
}

function gmailTimestamp() {
  const timeEl = document.querySelector(".g3");
  if (timeEl) return timeEl.getAttribute("title") || timeEl.innerText || new Date().toISOString();
  return new Date().toISOString();
}

function gmailSender() {
  const sender = document.querySelector(".gD");
  return sender ? (sender.getAttribute("email") || sender.innerText) : "unknown";
}

// ────────────────────────────────────────────────────────────────────────────
// OUTLOOK
// ────────────────────────────────────────────────────────────────────────────
function outlookConversationId() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get("ItemID") || params.get("conversationId");
  if (id) return `outlook_${id.slice(-16)}`;
  return `outlook_${Date.now()}`;
}

function outlookEmailText() {
  const body = document.querySelector("[class*='ReadingPane'] [class*='body']")
             || document.querySelector(".allowTextSelection");
  return body ? body.innerText.trim() : "";
}

function outlookTimestamp() {
  const timeEl = document.querySelector("[class*='DateTimeContainer']");
  return timeEl ? timeEl.innerText.trim() : new Date().toISOString();
}

// ────────────────────────────────────────────────────────────────────────────
// WHATSAPP WEB
// ────────────────────────────────────────────────────────────────────────────

/**
 * Contact name from the conversation header.
 * Multiple selector fallbacks — WA Web updates their DOM structure frequently.
 */
function waContactName() {
  const selectors = [
    "[data-testid='conversation-info-header-chat-title'] span",
    "[data-testid='conversation-info-header-chat-title']",
    "header [title]",
    "#main header span[dir='auto']",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el) {
      const name = (el.getAttribute("title") || el.innerText || "").trim();
      if (name) return name;
    }
  }
  // Fallback: document title — WA sets it to the contact name
  // Strip directional unicode marks WhatsApp sometimes injects
  const title = document.title
    .replace("WhatsApp", "")
    .replace(/[\u200e\u200f\u202a-\u202e]/g, "")
    .trim();
  return title || null;
}

/**
 * Stable conv_id from contact name using djb2 hash.
 * Same contact always produces the same ID — window accumulates correctly.
 * Phase 2: server resolves this via CNContactStore to a cross-platform entity_id.
 */
function waConversationId() {
  const name = waContactName();
  if (!name) return `wa_${Date.now()}`;
  let h = 5381;
  for (let i = 0; i < name.length; i++) {
    h = ((h << 5) + h) ^ name.charCodeAt(i);
    h = h >>> 0;  // keep unsigned 32-bit
  }
  return `wa_${h.toString(16).padStart(8, "0")}`;
}

/**
 * Most recent INCOMING message text.
 * Skips outgoing (.message-out) — user's own replies are not SE attacks.
 */
function waLastIncomingText() {
  const containers = document.querySelectorAll("[data-testid='msg-container']");
  for (let i = containers.length - 1; i >= 0; i--) {
    const c = containers[i];
    if (c.querySelector(".message-out") || c.classList.contains("message-out")) continue;
    const textEl = c.querySelector(".copyable-text");
    if (textEl) {
      const text = textEl.innerText.trim();
      if (text.length >= 10) return text;
    }
  }
  return "";
}

/**
 * User-selected text. Takes priority over auto-extracted last message.
 */
function waSelectedText() {
  const sel = window.getSelection();
  return sel && sel.toString().trim().length > 10 ? sel.toString().trim() : null;
}

/**
 * Timestamp from data-pre-plain-text attribute.
 * WA format: "[14:32, 12/05/2026] John Smith: "
 */
function waTimestamp() {
  const containers = document.querySelectorAll("[data-testid='msg-container']");
  for (let i = containers.length - 1; i >= 0; i--) {
    const c = containers[i];
    if (c.querySelector(".message-out")) continue;
    const pre = c.querySelector("[data-pre-plain-text]");
    if (pre) {
      const raw = pre.getAttribute("data-pre-plain-text") || "";
      const match = raw.match(/\[([\d:, \/]+)\]/);
      if (match) return `wa_ts_${match[1].replace(/[^\d]/g, "_")}`;
    }
  }
  return new Date().toISOString();
}

// ────────────────────────────────────────────────────────────────────────────
// UNIVERSAL EXTRACTOR
// ────────────────────────────────────────────────────────────────────────────
function extractEmailData() {
  const client = detectClient();
  let data = {
    client,
    conversation_id: `unknown_${Date.now()}`,
    text:            "",
    selected_text:   null,
    timestamp:       new Date().toISOString(),
    sender:          "unknown",
    sender_name:     null,   // Phase 2: server resolves via CNContactStore
    sender_email:    null,   // Phase 2: server resolves via CNContactStore
    url:             window.location.href,
  };

  if (client === "gmail") {
    data.conversation_id = gmailThreadId();
    data.text            = gmailEmailText();
    data.selected_text   = gmailSelectedText();
    data.timestamp       = gmailTimestamp();
    data.sender          = gmailSender();
    data.sender_email    = gmailSender();

  } else if (client.startsWith("outlook")) {
    data.conversation_id = outlookConversationId();
    data.text            = outlookEmailText();
    data.selected_text   = window.getSelection()?.toString().trim() || null;
    data.timestamp       = outlookTimestamp();

  } else if (client === "whatsapp_web") {
    data.conversation_id = waConversationId();
    data.text            = waLastIncomingText();
    data.selected_text   = waSelectedText();
    data.timestamp       = waTimestamp();
    data.sender          = waContactName() || "unknown";
    data.sender_name     = waContactName();
  }

  return data;
}

// ────────────────────────────────────────────────────────────────────────────
// MESSAGE LISTENER
// ────────────────────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  if (message.type === "EXTRACT_EMAIL") {
    const data       = extractEmailData();
    const textToScan = data.selected_text || data.text;

    if (!textToScan || textToScan.length < 10) {
      const hint = data.client === "whatsapp_web"
        ? "Open a WhatsApp conversation and select or receive a message first."
        : "Could not extract message text. Select the text manually and try again.";
      sendResponse({ success: false, error: hint, data: null });
      return true;
    }

    sendResponse({
      success: true,
      error:   null,
      data: {
        text:            textToScan,
        conversation_id: data.conversation_id,
        timestamp:       data.timestamp,
        sender:          data.sender,
        sender_name:     data.sender_name,
        sender_email:    data.sender_email,
        client:          data.client,
        is_selection:    !!data.selected_text,
      },
    });
    return true;
  }

  if (message.type === "PING") {
    sendResponse({ status: "ready", client: detectClient() });
    return true;
  }
});

console.log("[SE Shield v1.1.0] Content script ready on", detectClient());