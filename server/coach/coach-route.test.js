const assert = require("node:assert/strict");
const test = require("node:test");
const { Readable } = require("node:stream");
const { createCoachHandler } = require("./coach-route");

function makeRequest(body, url = "/api/coach") {
  const req = Readable.from([Buffer.from(JSON.stringify(body))]);
  req.headers = { "content-type": "application/json" };
  req.method = "POST";
  req.url = url;
  return req;
}

function makeResponse() {
  return {
    statusCode: 0,
    headers: null,
    body: "",
    writeHead(statusCode, headers) {
      this.statusCode = statusCode;
      this.headers = headers;
      return this;
    },
    end(chunk) {
      this.body = chunk || "";
      return this;
    },
  };
}

test("coach route returns remote advice when muse call succeeds", async () => {
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return { output_text: "Nice effort—curl your index finger and hold the shape steady." };
      },
    }),
  });

  const req = makeRequest({ target: "C", predicted: "O", errors: ["index_too_straight"] });
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.advice, /curl/i);
});

test("coach route falls back when muse call fails", async () => {
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    fetchImpl: async () => ({ ok: false, async json() { return {}; } }),
  });

  const req = makeRequest({ target: "C", predicted: "O", errors: ["thumb_too_far"] });
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.advice, /thumb|hand|shape/i);
});

test("coach route returns detailed advice with options", async () => {
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          output_text: "What I see: your index is too straight. Correction: curl the index and hold 2 seconds. Drill: 5 slow reps.",
        };
      },
    }),
  });

  const req = makeRequest({
    target: "C",
    predicted: "O",
    confidence: 0.51,
    holdMs: 600,
    errors: ["index_too_straight", "hold_too_short"],
    detailLevel: "detailed",
  });
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.advice, /index|curl|hold/i);
  assert.equal(Array.isArray(payload.options), true);
  assert.equal(payload.options.length > 0, true);
});

test("coach route supports chat replies", async () => {
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          output_text: "Great question. For letter C, curve all fingers together, keep thumb tucked, and hold 2 seconds before reset.",
        };
      },
    }),
  });

  const req = makeRequest({
    mode: "chat",
    message: "How do I improve letter C?",
    messages: [{ role: "assistant", content: "Let's work on C." }],
    context: { target: "C", predicted: "O", errors: ["thumb_too_far"] },
  }, "/api/coach/chat");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.reply, /C|curve|thumb|hold/i);
});

test("coach chat can attach a generated letter reference image when asked", async () => {
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAukB9U9KcbwAAAAASUVORK5CYII=";
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    imageEndpoint: "https://example.com/muse-image",
    imageApiKey: "test-key",
    visionEndpoint: "https://example.com/muse-vision",
    visionApiKey: "test-key",
    fetchImpl: async (url, options) => {
      if (String(url).includes("muse-image")) {
        return {
          ok: true,
          async json() {
            return { data: [{ b64_json: tinyPngBase64 }] };
          },
        };
      }
      if (String(url).includes("muse-vision")) {
        const body = JSON.parse(options?.body || "{}");
        const promptText = JSON.stringify(body?.messages || []);
        if (/Return ONLY YES or NO/i.test(promptText)) {
          if (/letter C\?/i.test(promptText)) {
            return {
              ok: true,
              async json() {
                return { output_text: "YES" };
              },
            };
          }
          return {
            ok: true,
            async json() {
              return { output_text: "NO" };
            },
          };
        }
        return {
          ok: true,
          async json() {
            return { output_text: "C" };
          },
        };
      }
      return {
        ok: true,
        async json() {
          return { output_text: "Sure—here’s a reference plus a quick correction." };
        },
      };
    },
  });

  const req = makeRequest({
    mode: "chat",
    message: "Can you show me what letter C should look like?",
    context: { target: "C" },
  }, "/api/coach/chat");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.reply, /reference|correction|sure/i);
  assert.equal(payload.image?.letter, "C");
  assert.match(payload.image?.imageDataUrl || "", /^data:image\/png;base64,/);
  assert.equal(payload.image?.confusableDetected || null, null);
});

test("coach image endpoint returns a generated reference for one letter", async () => {
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAukB9U9KcbwAAAAASUVORK5CYII=";
  const handler = createCoachHandler({
    imageEndpoint: "https://example.com/muse-image",
    imageApiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return { data: [{ b64_json: tinyPngBase64 }] };
      },
    }),
  });

  const req = makeRequest({ letter: "a" }, "/api/coach/image");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.equal(payload.image.letter, "A");
  assert.match(payload.image.imageDataUrl, /^data:image\/png;base64,/);
});

test("coach image endpoint builds strict prompt guidance for letter N", async () => {
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAukB9U9KcbwAAAAASUVORK5CYII=";
  const handler = createCoachHandler({
    imageEndpoint: "https://example.com/muse-image",
    imageApiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return { data: [{ b64_json: tinyPngBase64 }] };
      },
    }),
  });

  const req = makeRequest({ letter: "n" }, "/api/coach/image");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.equal(payload.image.letter, "N");
  assert.match(payload.image.prompt, /two finger knuckles visible over thumb/i);
  assert.match(payload.image.prompt, /Not M/i);
});

test("coach image endpoint retries when verification detects the wrong letter", async () => {
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAukB9U9KcbwAAAAASUVORK5CYII=";
  let identifyCalls = 0;
  let binaryCalls = 0;
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    imageEndpoint: "https://example.com/muse-image",
    imageApiKey: "test-key",
    visionEndpoint: "https://example.com/muse-vision",
    visionApiKey: "test-key",
    fetchImpl: async (url, options) => {
      if (String(url).includes("muse-image")) {
        return {
          ok: true,
          async json() {
            return { data: [{ b64_json: tinyPngBase64 }] };
          },
        };
      }
      if (String(url).includes("muse-vision")) {
        const body = JSON.parse(options?.body || "{}");
        const promptText = JSON.stringify(body?.messages || []);
        if (/Return ONLY YES or NO/i.test(promptText)) {
          binaryCalls += 1;
          if (/letter N\?/i.test(promptText)) {
            return {
              ok: true,
              async json() {
                return { output_text: binaryCalls === 1 ? "NO" : "YES" };
              },
            };
          }
          return {
            ok: true,
            async json() {
              return { output_text: "NO" };
            },
          };
        }
        identifyCalls += 1;
        return {
          ok: true,
          async json() {
            return { output_text: identifyCalls === 1 ? "M" : "N" };
          },
        };
      }
      return {
        ok: true,
        async json() {
          return { output_text: "N" };
        },
      };
    },
  });

  const req = makeRequest({ letter: "N" }, "/api/coach/image");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.equal(payload.image.letter, "N");
  assert.equal(payload.image.verified, true);
  assert.equal(payload.image.detectedLetter, "N");
  assert.equal(payload.image.targetConfirmed, true);
  assert.equal(payload.image.confusableDetected, null);
  assert.equal(payload.image.attempts, 2);
});

test("coach image endpoint returns best-effort output when verification never confirms target", async () => {
  const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAukB9U9KcbwAAAAASUVORK5CYII=";
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    apiKey: "test-key",
    imageEndpoint: "https://example.com/muse-image",
    imageApiKey: "test-key",
    visionEndpoint: "https://example.com/muse-vision",
    visionApiKey: "test-key",
    imageVerificationMaxAttempts: 2,
    fetchImpl: async (url, options) => {
      if (String(url).includes("muse-image")) {
        return {
          ok: true,
          async json() {
            return { data: [{ b64_json: tinyPngBase64 }] };
          },
        };
      }
      if (String(url).includes("muse-vision")) {
        const body = JSON.parse(options?.body || "{}");
        const promptText = JSON.stringify(body?.messages || []);
        if (/Return ONLY YES or NO/i.test(promptText)) {
          return {
            ok: true,
            async json() {
              return { output_text: "NO" };
            },
          };
        }
        return {
          ok: true,
          async json() {
            return { output_text: "M" };
          },
        };
      }
      return {
        ok: true,
        async json() {
          return { output_text: "M" };
        },
      };
    },
  });

  const req = makeRequest({ letter: "N" }, "/api/coach/image");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.equal(payload.image.letter, "N");
  assert.equal(payload.image.verified, false);
  assert.match(payload.image.imageDataUrl, /^data:image\/png;base64,/);
});

test("coach chat supports photo attachment payload", async () => {
  const handler = createCoachHandler({
    endpoint: "https://example.com/muse-text",
    visionEndpoint: "https://example.com/muse-vision",
    apiKey: "test-key",
    visionApiKey: "test-key",
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          output_text: "Great framing. Curl your index a bit more, tuck your thumb, and hold two seconds.",
        };
      },
    }),
  });

  const req = makeRequest({
    mode: "chat",
    message: "Can you review my hand photo?",
    imageDataUrl: "data:image/png;base64,AA==",
    context: { target: "C", predicted: "O", errors: ["thumb_too_far"] },
  }, "/api/coach/chat");
  const res = makeResponse();
  await handler(req, res);

  assert.equal(res.statusCode, 200);
  const payload = JSON.parse(res.body);
  assert.match(payload.reply, /curl|thumb|hold|framing/i);
});
