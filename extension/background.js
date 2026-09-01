/**
 * extension/background.js
 * Service worker - handles server health check and badge updates
 */
"use strict";

const SERVER = "http://127.0.0.1:8000";

// Check server health on install / startup
async function checkServer() {
  try {
    const r = await fetch(`${SERVER}/health`, { method: "GET" });
    if (r.ok) {
      const data = await r.json();
      chrome.action.setBadgeBackgroundColor({ color: "#00C9A7" });
      chrome.action.setBadgeText({ text: "" });
      console.log("[SE Shield] Server ready:", data);
    }
  } catch (e) {
    chrome.action.setBadgeBackgroundColor({ color: "#E05252" });
    chrome.action.setBadgeText({ text: "!" });
    console.warn("[SE Shield] Server not reachable:", e.message);
  }
}

// Update badge based on alert level
function setBadge(alertLevel) {
  const map = {
    CRITICAL: { color: "#E05252", text: "!!" },
    HIGH:     { color: "#E05252", text: "!"  },
    MEDIUM:   { color: "#F0943A", text: "~"  },
    LOW:      { color: "#00C9A7", text: ""   },
  };
  const b = map[alertLevel] || map.LOW;
  chrome.action.setBadgeBackgroundColor({ color: b.color });
  chrome.action.setBadgeText({ text: b.text });
}

// Listen for alert level updates from popup
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "UPDATE_BADGE") setBadge(msg.alert_level);
});

chrome.runtime.onInstalled.addListener(checkServer);
chrome.runtime.onStartup.addListener(checkServer);