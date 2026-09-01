/**
 * extension/details.js
 * Full breakdown window — reads stored scan result from chrome.storage.session
 * and renders all layers: L2 risk, L3 labels+probs, L4a context, L4b assessment.
 */
"use strict";

const LABEL_DISPLAY = {
  phishing:                  "Phishing",
  spear_phishing:            "Spear Phishing",
  pretexting:                "Pretexting",
  credential_harvesting:     "Credential Harvesting",
  baiting:                   "Baiting",
  vishing:                   "Vishing",
  business_email_compromise: "BEC",
  benign:                    "Benign",
};

function rc(risk) {
  return risk >= 60 ? "var(--red)" : risk >= 35 ? "var(--orange)" : "var(--green)";
}
function rClass(risk) {
  return risk >= 60 ? "red" : risk >= 35 ? "orange" : "green";
}
function pill(val) {
  return `<span class="pill ${val}">${val ? "YES" : "NO"}</span>`;
}
function probClass(i) {
  return i === 0 ? "top" : i <= 2 ? "mid" : "low";
}
function lbl(key) {
  return LABEL_DISPLAY[key] || key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ── Card builder helpers ──────────────────────────────────────────────────

function card(layerTag, title, bodyHTML) {
  return `
    <div class="card">
      <div class="card-header">
        <span class="card-label">${title}</span>
        <span class="layer-badge">${layerTag}</span>
      </div>
      <div class="card-body">${bodyHTML}</div>
    </div>`;
}

function kv(key, valHTML) {
  return `<div class="kv"><span class="k">${key}</span><span>${valHTML}</span></div>`;
}

function meterBar(value, max, color) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  return `
    <div class="meter-row">
      <div class="meter-track">
        <div class="meter-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <div class="meter-val" style="color:${color}">${value}</div>
    </div>`;
}

// ── Section renderers ─────────────────────────────────────────────────────

function renderConversationHero(data) {
  const level   = data.alert_level   || "LOW";
  const risk    = data.entity_risk   || 0;
  const pattern = (data.attack_pattern || "none").replace(/_/g, " ").toUpperCase();
  const color   = rc(risk);

  return `
    <div class="card">
      <div class="card-header">
        <span class="card-label">Conversation Assessment</span>
        <span class="layer-badge">Layer 4b</span>
      </div>
      <div class="alert-hero ${level}">
        <div class="alert-left">
          <div class="alert-level-txt ${level}">${level}</div>
          <div class="alert-pattern">${pattern}</div>
          <div style="margin-top:6px;font-size:11px;color:var(--muted)">
            ${data.window_size || 0} message${data.window_size !== 1 ? "s" : ""} in window
            &nbsp;·&nbsp;
            dominant: <span style="color:var(--white)">${lbl(data.dominant_label || "none")}</span>
          </div>
        </div>
        <div class="risk-arc-wrap">
          <div class="risk-arc-val" style="color:${color}">${risk}</div>
          <div class="risk-arc-lbl">/ 100 risk</div>
          <div style="margin-top:6px;width:80px">
            <div class="meter-track">
              <div class="meter-fill" style="width:${risk}%;background:${color}"></div>
            </div>
          </div>
        </div>
      </div>
    </div>`;
}

function renderL3(last) {
  if (!last || !last.label) return "";

  const probs = last.probabilities || {};
  const topLabel = last.label;
  const topConf  = last.confidence || 0;

  // Sort by confidence descending
  const sorted = Object.entries(probs).sort((a,b) => b[1] - a[1]);

  const probBars = sorted.map(([key, val], i) => {
    const isTop = key === topLabel;
    const pct   = (val * 100).toFixed(1);
    const cls   = probClass(i);
    return `
      <div class="prob-row">
        <div class="prob-label ${isTop ? 'top' : ''}">${lbl(key)}</div>
        <div class="prob-track">
          <div class="prob-fill ${cls}" style="width:${pct}%"></div>
        </div>
        <div class="prob-pct ${isTop ? 'top' : ''}">${pct}%</div>
      </div>`;
  }).join("");

  const body = `
    <div class="top-label-row">
      <span class="top-label-name">${lbl(topLabel)}</span>
      <span class="top-label-conf">${(topConf * 100).toFixed(1)}%</span>
    </div>
    <div class="divider"></div>
    <div class="kv"><span class="k">Reason</span></div>
    <div style="font-size:11px;color:var(--body);line-height:1.6;padding:2px 0">
      ${(last.reason || "—").replace(/\(low confidence.*?\)/g, s => `<span style="color:var(--muted);font-size:10px">${s}</span>`)}
    </div>
    <div class="divider"></div>
    <div class="card-label" style="margin-bottom:8px">All Label Probabilities</div>
    <div class="prob-list">${probBars}</div>`;

  return card("Layer 3 — NLI", "Message Classification", body);
}

function renderL2(last) {
  if (!last) return "";
  const risk  = last.layer2_risk || 0;
  const color = rc(risk);
  const lat   = last.latency_ms != null ? `${last.latency_ms.toFixed(0)} ms` : "—";

  const body = `
    ${kv("SVM Gate", `<span class="v ${risk > 0 ? 'red' : 'green'}">${risk > 0 ? "SUSPICIOUS" : "BENIGN"}</span>`)}
    ${kv("LR Risk Score", `<span class="v ${rClass(risk)}" style="font-family:'Space Mono',monospace">${risk}/100</span>`)}
    ${meterBar(risk, 100, color)}
    ${kv("NLI Latency", `<span class="v blue">${lat}</span>`)}`;

  return card("Layer 2 + 3", "ML Triage Signals", body);
}

function renderL4a(ctx) {
  if (!ctx) return "";
  const acc   = (ctx.accumulated_risk || 0).toFixed(3);
  const sus   = ctx.suspicious_flag || false;
  const ovr   = ctx.svm_override    || false;
  const msgs  = ctx.messages_seen   || 0;

  const body = `
    ${kv("Accumulated Risk", `<span class="v ${sus ? 'red' : 'accent'}" style="font-family:'Space Mono',monospace">${acc}</span>`)}
    ${kv("Suspicious Flag",  pill(sus))}
    ${kv("SVM Override",     pill(ovr))}
    ${kv("Messages Seen",    `<span class="v blue">${msgs}</span>`)}
    ${meterBar(Math.min(10, parseFloat(acc)), 10, sus ? "var(--red)" : "var(--accent)")}`;

  return card("Layer 4a", "Risk Counter (Conversation Context)", body);
}

function renderReasons(reasons) {
  if (!reasons || !reasons.length) return "";

  const items = reasons.map(r => {
    const isBenign = r.toLowerCase().includes("benign");
    return `<div class="reason-item ${isBenign ? 'benign' : 'attack'}">${r}</div>`;
  }).join("");

  return card("Layer 4b", "Reasoning Trail", `<div style="display:flex;flex-direction:column;gap:7px">${items}</div>`);
}

function renderMeta(data) {
  const last    = data.last_message || {};
  const msgId   = last.message_id  || "—";
  const ts      = last.timestamp   || "—";
  const convId  = data.conversation_id || "—";
  const winSz   = data.window_size  || 0;
  const scanned = data.scanned_at   || "—";

  const body = `
    ${kv("Conversation ID",  `<span class="v muted" style="word-break:break-all;max-width:280px">${convId}</span>`)}
    ${kv("Message ID",       `<span class="v muted">${msgId}</span>`)}
    ${kv("Timestamp",        `<span class="v muted">${ts}</span>`)}
    ${kv("Window Size",      `<span class="v blue">${winSz}</span>`)}
    ${kv("Scanned At",       `<span class="v muted">${scanned}</span>`)}`;

  return card("Meta", "Scan Metadata", body);
}

// ── Main render ───────────────────────────────────────────────────────────

function render(data) {
  const noData  = document.getElementById("noData");
  const content = document.getElementById("mainContent");
  const hdr     = document.getElementById("convIdHdr");

  if (!data) {
    noData.style.display = "block";
    return;
  }

  noData.style.display = "none";

  const conv = data.conversation_id || "";
  hdr.textContent = conv ? `conv: ${conv.slice(0, 28)}${conv.length > 28 ? "…" : ""}` : "—";

  const last = data.last_message || {};
  const ctx  = data.conversation_context || null;

  content.innerHTML =
    renderConversationHero(data)  +
    renderL3(last)                +
    renderL2(last)                +
    renderL4a(ctx)                +
    renderReasons(data.reasons)   +
    renderMeta(data);
}

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.session.get(["lastResult"], (stored) => {
    render(stored.lastResult || null);
  });

  // Live update if a new scan happens while window is open
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "session" && changes.lastResult) {
      render(changes.lastResult.newValue || null);
    }
  });
});