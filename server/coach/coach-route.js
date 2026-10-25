const {
  buildCoachMessages,
  buildDetailedCoachMessages,
  buildCoachChatMessages,
  buildCoachPhotoMessages,
  buildTutorialLessonMessages,
  buildTutorialFollowUpMessages,
  buildLetterImagePrompt,
  buildLetterImageRetryPrompt,
  buildLetterImageVerificationMessages,
  buildLetterImageTargetCheckMessages,
  sanitizeAdvice,
  sanitizeDetailedAdvice,
} = require("./coach-prompt");
const { fallbackAdvice } = require("./coach-fallback");
const { getConfusableLetters } = require("./letter-rules");

function parseJsonBody(request, maxBytes = 16 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    let raw = "";
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        reject(Object.assign(new Error("Payload too large"), { statusCode: 413 }));
        request.destroy();
        return;
      }
      raw += chunk.toString("utf8");
    });
    request.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        reject(Object.assign(new Error("Invalid JSON"), { statusCode: 400 }));
      }
    });
    request.on("error", reject);
  });
}

function buildFallbackOptions(payload) {
  const errors = Array.isArray(payload?.errors) ? payload.errors : [];
  const options = [];
  if (errors.includes("hold_too_short")) options.push("Try 3 slow holds: hold each shape for a full 2 seconds.");
  if (errors.includes("index_too_straight")) options.push("Practice curling your index finger 10 times before signing again.");
  if (errors.includes("fingers_not_together")) options.push("Tap fingertips together lightly, then form the target shape.");
  if (errors.includes("thumb_too_far")) options.push("Keep your thumb close to your index knuckle while holding the sign.");
  if (!options.length) options.push("Do a 30-second slow-motion drill: shape, hold, relax, and repeat.");
  return options.slice(0, 3);
}

function fallbackDetailedAdvice(payload) {
  const target = String(payload?.target || "?").toUpperCase();
  const predicted = String(payload?.predicted || "?").toUpperCase();
  const base = fallbackAdvice(payload?.errors);
  const options = buildFallbackOptions(payload);
  return {
    advice: [
      `You’re practicing ${target}, and the model is currently seeing ${predicted}.`,
      base,
      `Try this drill: ${options[0]}`,
      "Confidence tip: prioritize a steady hold over speed.",
    ].join("\n"),
    options,
  };
}

function fallbackChatReply(payload) {
  const message = String(payload?.message || "").trim();
  const hint = fallbackAdvice(payload?.context?.errors);
  if (!message) {
    return "I can help! Tell me which letter you’re practicing, and I’ll give step-by-step corrections plus a short drill.";
  }
  return [
    "Great question.",
    hint,
    "If you want, I can break it into 3 tiny steps and watch for your next attempt.",
  ].join(" ");
}

function parseRequestedLetter(message = "", context = {}) {
  const direct = String(context?.target || "").trim().toUpperCase();
  if (/^[A-Z]$/.test(direct) && /(show|image|picture|look|reference|example|what.*look)/i.test(message)) {
    return direct;
  }
  const pattern = /\bLETTER\s+([A-Z])\b|\b([A-Z])\b/;
  const match = pattern.exec(String(message).toUpperCase());
  if (!match) return null;
  const letter = match[1] || match[2];
  if (!/^[A-Z]$/.test(letter)) return null;
  if (!/(show|image|picture|look|reference|example|what.*look|generate)/i.test(message)) return null;
  return letter;
}

function dataUrlImageType(dataUrl) {
  const match = /^data:(image\/(png|jpeg|jpg|webp));base64,/i.exec(String(dataUrl || ""));
  return match ? match[1].toLowerCase().replace("jpg", "jpeg") : null;
}

function fallbackPhotoAdvice(payload) {
  const target = String(payload?.context?.target || payload?.target || "?").toUpperCase();
  const hint = fallbackAdvice(payload?.context?.errors || payload?.errors || []);
  return [
    `Nice effort on ${target}. I can still help even when photo analysis is unavailable.`,
    hint,
    "Try this: hold the shape 2 seconds, relax, then repeat 5 times while keeping your wrist centered.",
  ].join("\n");
}

function fallbackTutorialLesson(payload) {
  const letter = String(payload?.target || payload?.letter || "A").slice(0, 1).toUpperCase();
  const focus = String(payload?.focus || "shape").toLowerCase();
  if (focus === "mistakes") {
    return [
      `Mistake 1 for ${letter}: over-straightening fingers — curl to a natural bend before placing thumb.`,
      "Mistake 2: thumb floats too far out — anchor thumb to the correct finger contact point.",
      "Mistake 3: palm rotates mid-hold — lock wrist and keep one stable palm angle.",
      "After each correction, rebuild from knuckles instead of tweaking mid-shape.",
      "Retry drill: 3 rounds of 2 reps; hold each rep 2 seconds and reset fully between reps.",
      "Pass when all 3 reps keep thumb position and palm angle consistent.",
    ].join("\n");
  }
  if (focus === "drill") {
    return [
      `0-15s: Build letter ${letter} slowly, 2 reps, 2-second holds, full reset each rep.`,
      "15-30s: Increase pace to 3 reps while preserving exact thumb and finger placement.",
      "30-45s: Alternate eyes-open and mirror-check reps to catch palm-angle drift.",
      "45-60s: Final 2 clean reps with no corrections mid-hold.",
      "Success check: complete 7 total reps with stable palm orientation and no thumb drift.",
    ].join("\n");
  }
  return [
    `Set letter ${letter} with a relaxed wrist and steady hand at chest height.`,
    "Step 1: place base finger shape before moving thumb.",
    "Step 2: set thumb contact point with a small, controlled motion.",
    "Step 3: rotate palm to final orientation and hold still for 2 seconds.",
    "Step 4: check fingertips, knuckles, thumb, then wrist alignment in that order.",
    "Self-check: no finger gaps and no palm rotation during hold.",
    "Drill: 6 reps total — hold each rep for 2 seconds, rest 2 seconds between reps.",
  ].join("\n");
}

function fallbackTutorialFollowUp(payload) {
  const letter = String(payload?.target || payload?.letter || "A").slice(0, 1).toUpperCase();
  const focus = String(payload?.focus || "shape").toLowerCase();
  let focusLabel = "hand shape";
  if (focus === "mistakes") focusLabel = "common mistakes";
  if (focus === "drill") focusLabel = "drill timing";
  return [
    `I'm having trouble reaching Muse right now for letter ${letter}.`,
    `Quick ${focusLabel} fallback: hold the target shape for 2 seconds, reset fully, repeat 6 reps.`,
    "Try again in a moment for a full AI follow-up answer.",
  ].join(" ");
}

function extractSingleLetter(text = "") {
  const match = /\b([A-Z])\b/.exec(String(text).toUpperCase());
  return match ? match[1] : null;
}

function createCoachHandler(options = {}) {
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  const imageFetchImpl = options.imageFetchImpl || fetchImpl;
  const endpoint = options.endpoint;
  const apiKey = options.apiKey;
  const model = options.model || "muse-text-1.0";
  const imageEndpoint = options.imageEndpoint;
  const imageApiKey = options.imageApiKey || apiKey;
  const imageModel = options.imageModel || "muse-image-1.0";
  const imageSize = options.imageSize || "512x512";
  const visionEndpoint = options.visionEndpoint || endpoint;
  const visionApiKey = options.visionApiKey || apiKey;
  const visionModel = options.visionModel || model;
  const imageVerificationMaxAttempts = Number(options.imageVerificationMaxAttempts || 3);
  const hasVisionVerifier = Boolean(visionEndpoint && visionApiKey);

  async function callMuseImage(prompt) {
    if (!imageFetchImpl || !imageEndpoint || !imageApiKey) return null;
    const response = await imageFetchImpl(imageEndpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${imageApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: imageModel,
        prompt,
        n: 1,
        size: imageSize,
        response_format: "b64_json",
        output_format: "png",
      }),
    });
    if (!response.ok) return null;
    const data = await response.json();
    const encoded = data?.data?.[0]?.b64_json;
    if (!encoded || typeof encoded !== "string") return null;
    return `data:image/png;base64,${encoded}`;
  }

  async function callMuse(messages, temperature = 0.2, callOptions = {}) {
    const effectiveEndpoint = callOptions.endpoint || endpoint;
    const effectiveApiKey = callOptions.apiKey || apiKey;
    const effectiveModel = callOptions.model || model;
    if (!fetchImpl || !effectiveEndpoint || !effectiveApiKey) return null;
    const response = await fetchImpl(effectiveEndpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${effectiveApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: effectiveModel,
        messages,
        temperature,
      }),
    });
    if (!response.ok) return null;
    const data = await response.json();
    return data?.output_text || data?.choices?.[0]?.message?.content || "";
  }

  async function verifyGeneratedLetterImage(imageDataUrl, targetLetter) {
    if (!imageDataUrl) return null;
    if (!hasVisionVerifier) {
      return {
        accepted: false,
        detectedLetter: null,
        targetConfirmed: null,
        unavailable: true,
      };
    }
    const identifyText = await callMuse(
      buildLetterImageVerificationMessages(targetLetter, imageDataUrl),
      0,
      { endpoint: visionEndpoint, apiKey: visionApiKey, model: visionModel },
    );
    const checkText = await callMuse(
      buildLetterImageTargetCheckMessages(targetLetter, imageDataUrl),
      0,
      { endpoint: visionEndpoint, apiKey: visionApiKey, model: visionModel },
    );
    const detectedLetter = extractSingleLetter(identifyText) || "UNKNOWN";
    const targetConfirmed = /\bYES\b/i.test(String(checkText || ""));

    let confusableDetected = null;
    if (detectedLetter === targetLetter && targetConfirmed) {
      const confusables = getConfusableLetters(targetLetter).slice(0, 3);
      for (const confusableLetter of confusables) {
        const confusableCheckText = await callMuse(
          buildLetterImageTargetCheckMessages(confusableLetter, imageDataUrl),
          0,
          { endpoint: visionEndpoint, apiKey: visionApiKey, model: visionModel },
        );
        const confusableMatched = /\bYES\b/i.test(String(confusableCheckText || ""));
        if (confusableMatched) {
          confusableDetected = confusableLetter;
          break;
        }
      }
    }

    return {
      accepted: detectedLetter === targetLetter && targetConfirmed && !confusableDetected,
      detectedLetter,
      targetConfirmed,
      confusableDetected,
      unavailable: false,
    };
  }

  async function generateVerifiedLetterImage(letter) {
    let firstImageDataUrl = null;
    let mismatchLetter = "";
    const maxAttempts = Number.isFinite(imageVerificationMaxAttempts)
      ? Math.max(1, Math.min(5, imageVerificationMaxAttempts))
      : 3;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const prompt = attempt === 1
        ? buildLetterImagePrompt(letter)
        : buildLetterImageRetryPrompt(letter, mismatchLetter);
      const imageDataUrl = await callMuseImage(prompt);
      if (!imageDataUrl) continue;
      if (!firstImageDataUrl) firstImageDataUrl = imageDataUrl;

      const verification = await verifyGeneratedLetterImage(imageDataUrl, letter);
      if (verification?.accepted) {
        return {
          imageDataUrl,
          prompt,
          verified: true,
          detectedLetter: verification.detectedLetter || null,
          targetConfirmed: Boolean(verification.targetConfirmed),
          confusableDetected: verification.confusableDetected || null,
          attempts: attempt,
        };
      }
      if (verification?.unavailable) {
        return {
          imageDataUrl,
          prompt,
          verified: false,
          detectedLetter: null,
          targetConfirmed: null,
          confusableDetected: null,
          attempts: attempt,
          verificationUnavailable: true,
        };
      }
      mismatchLetter = verification?.confusableDetected || verification?.detectedLetter || "";
    }

    if (hasVisionVerifier) {
      return {
        imageDataUrl: firstImageDataUrl,
        prompt: buildLetterImageRetryPrompt(letter, mismatchLetter),
        verified: false,
        detectedLetter: mismatchLetter || null,
        targetConfirmed: false,
        confusableDetected: mismatchLetter || null,
        attempts: maxAttempts,
        bestEffort: Boolean(firstImageDataUrl),
      };
    }

    return {
      imageDataUrl: firstImageDataUrl,
      prompt: buildLetterImageRetryPrompt(letter, mismatchLetter),
      verified: false,
      detectedLetter: mismatchLetter || null,
      targetConfirmed: null,
      confusableDetected: mismatchLetter || null,
      attempts: maxAttempts,
    };
  }

  async function generateAdvice(payload) {
    const fallback = fallbackAdvice(payload?.errors);
    try {
      const text = await callMuse(buildCoachMessages(payload), 0.2);
      if (!text) return { advice: fallback, options: buildFallbackOptions(payload), source: "fallback" };
      return {
        advice: sanitizeAdvice(text, fallback),
        options: buildFallbackOptions(payload),
        source: "remote",
      };
    } catch {
      return { advice: fallback, options: buildFallbackOptions(payload), source: "fallback" };
    }
  }

  async function generateDetailedAdvice(payload) {
    const fallback = fallbackDetailedAdvice(payload);
    try {
      const text = await callMuse(buildDetailedCoachMessages(payload), 0.35);
      if (!text) return { ...fallback, source: "fallback" };
      return {
        advice: sanitizeDetailedAdvice(text, fallback.advice),
        options: buildFallbackOptions(payload),
        source: "remote",
      };
    } catch {
      return { ...fallback, source: "fallback" };
    }
  }

  async function generateChatReply(payload) {
    const fallback = fallbackChatReply(payload);
    const message = String(payload?.message || "");
    const imageRequestedLetter = parseRequestedLetter(message, payload?.context);
    const imageDataUrl = String(payload?.imageDataUrl || "");
    const hasPhoto = Boolean(dataUrlImageType(imageDataUrl));
    const fallbackReply = hasPhoto ? fallbackPhotoAdvice(payload) : fallback;
    try {
      const messages = hasPhoto
        ? buildCoachPhotoMessages(payload)
        : buildCoachChatMessages(payload);
      const text = await callMuse(messages, hasPhoto ? 0.35 : 0.45, hasPhoto
        ? { endpoint: visionEndpoint, apiKey: visionApiKey, model: visionModel }
        : {});
      const reply = text ? sanitizeDetailedAdvice(text, fallback) : fallbackReply;
      const response = {
        reply,
        source: text ? "remote" : "fallback",
      };
      if (!imageRequestedLetter) return response;
      const generated = await generateVerifiedLetterImage(imageRequestedLetter);
      if (!generated?.imageDataUrl) return response;
      return {
        ...response,
        image: {
          type: "letter-reference",
          letter: imageRequestedLetter,
          prompt: generated.prompt,
          imageDataUrl: generated.imageDataUrl,
          verified: generated.verified,
          detectedLetter: generated.detectedLetter,
          targetConfirmed: generated.targetConfirmed,
          confusableDetected: generated.confusableDetected,
          attempts: generated.attempts,
        },
      };
    } catch {
      return { reply: fallbackReply, source: "fallback" };
    }
  }

  async function generateTutorialLesson(payload) {
    const fallback = fallbackTutorialLesson(payload);
    try {
      const text = await callMuse(buildTutorialLessonMessages(payload), 0.3);
      const lesson = text ? sanitizeDetailedAdvice(text, fallback) : fallback;
      return {
        lesson,
        source: text ? "remote" : "fallback",
      };
    } catch {
      return { lesson: fallback, source: "fallback" };
    }
  }

  async function generateTutorialFollowUp(payload) {
    const fallback = fallbackTutorialFollowUp(payload);
    const question = String(payload?.question || payload?.message || "").trim();
    if (!question) {
      return {
        reply: "Please enter a follow-up question.",
        source: "fallback",
      };
    }
    try {
      const text = await callMuse(buildTutorialFollowUpMessages(payload), 0.45);
      const reply = text ? sanitizeDetailedAdvice(text, fallback) : fallback;
      return {
        reply,
        source: text ? "remote" : "fallback",
      };
    } catch {
      return { reply: fallback, source: "fallback" };
    }
  }

  async function generateLetterImage(payload) {
    const letter = String(payload?.letter || "").trim().toUpperCase();
    if (!/^[A-Z]$/.test(letter)) {
      return { error: "Send one letter A-Z.", statusCode: 400 };
    }
    const prompt = buildLetterImagePrompt(letter);
    try {
      const generated = await generateVerifiedLetterImage(letter);
      if (!generated?.imageDataUrl) {
        return {
          error: "Image generation is unavailable right now.",
          statusCode: 503,
        };
      }
      return {
        image: {
          type: "letter-reference",
          letter,
          prompt: generated.prompt || prompt,
          imageDataUrl: generated.imageDataUrl,
          verified: generated.verified,
          detectedLetter: generated.detectedLetter,
          targetConfirmed: generated.targetConfirmed,
          confusableDetected: generated.confusableDetected,
          attempts: generated.attempts,
        },
        source: "remote",
      };
    } catch {
      return {
        error: "Image generation is unavailable right now.",
        statusCode: 503,
      };
    }
  }

  return async function handleCoachRequest(request, response) {
    try {
      const isChat = request.url?.startsWith("/api/coach/chat");
      const isImage = request.url?.startsWith("/api/coach/image");
      const payload = await parseJsonBody(request, isChat ? 3 * 1024 * 1024 : 64 * 1024);
      let body;
      if (isImage || payload.mode === "image") {
        body = await generateLetterImage(payload);
        if (body.error) {
          response.writeHead(body.statusCode || 400, { "Content-Type": "application/json; charset=utf-8" });
          response.end(JSON.stringify({ error: body.error }));
          return;
        }
      } else if (payload.mode === "tutorial-lesson") {
        body = await generateTutorialLesson(payload);
      } else if (payload.mode === "tutorial-followup") {
        body = await generateTutorialFollowUp(payload);
      } else if (isChat || payload.mode === "chat") {
        body = await generateChatReply(payload);
      } else if (payload.detailLevel === "detailed") {
        body = await generateDetailedAdvice(payload);
      } else {
        body = await generateAdvice(payload);
      }
      response.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      response.end(JSON.stringify(body));
    } catch (error) {
      const status = error.statusCode || 500;
      response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
      response.end(JSON.stringify({ error: status === 500 ? "Internal error" : error.message }));
    }
  };
}

module.exports = { createCoachHandler, parseJsonBody };
