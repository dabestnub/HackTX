const { ERROR_HINTS } = require("./landmark-rules");

function createAdviceClient(options = {}) {
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  const endpoint = options.endpoint || "/api/coach";
  const minIntervalMs = Number(options.minIntervalMs || 2000);
  const maxWords = Number(options.maxWords || 18);
  const now = options.now || (() => Date.now());

  const cache = new Map();
  let lastRequestedAt = 0;

  function signature(payload) {
    const errors = Array.isArray(payload?.errors) ? [...payload.errors].sort().join(",") : "";
    return `${payload?.target || ""}|${payload?.predicted || ""}|${errors}`;
  }

  function fallbackAdvice(payload) {
    const first = payload?.errors?.[0];
    if (first && ERROR_HINTS[first]) return ERROR_HINTS[first];
    return "Nice try. Keep your hand steady and match the target shape.";
  }

  function sanitize(text) {
    if (!text || typeof text !== "string") return "";
    const words = text.trim().split(/\s+/).slice(0, maxWords);
    return words.join(" ");
  }

  async function requestAdvice(payload) {
    const key = signature(payload);
    if (cache.has(key)) return { advice: cache.get(key), source: "cache" };

    const t = now();
    if (t - lastRequestedAt < minIntervalMs) {
      return { advice: fallbackAdvice(payload), source: "throttle" };
    }
    lastRequestedAt = t;

    if (typeof fetchImpl !== "function") {
      return { advice: fallbackAdvice(payload), source: "fallback" };
    }

    try {
      const response = await fetchImpl(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) return { advice: fallbackAdvice(payload), source: "fallback" };
      const body = await response.json();
      const advice = sanitize(body?.advice) || fallbackAdvice(payload);
      cache.set(key, advice);
      return { advice, source: "remote" };
    } catch {
      return { advice: fallbackAdvice(payload), source: "fallback" };
    }
  }

  return { requestAdvice, _cache: cache };
}

module.exports = { createAdviceClient };
