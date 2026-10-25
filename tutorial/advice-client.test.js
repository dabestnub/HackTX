const assert = require("node:assert/strict");
const test = require("node:test");
const { createAdviceClient } = require("./advice-client");

test("advice client caches remote responses", async () => {
  let calls = 0;
  const client = createAdviceClient({
    endpoint: "/api/coach",
    minIntervalMs: 0,
    fetchImpl: async () => {
      calls += 1;
      return {
        ok: true,
        async json() {
          return { advice: "Great effort, curl your index finger more and keep your hand steady." };
        },
      };
    },
  });

  const payload = { target: "C", predicted: "O", errors: ["index_too_straight"] };
  const first = await client.requestAdvice(payload);
  const second = await client.requestAdvice(payload);
  assert.equal(first.source, "remote");
  assert.equal(second.source, "cache");
  assert.equal(calls, 1);
});

test("advice client throttles and falls back", async () => {
  let now = 0;
  const client = createAdviceClient({
    minIntervalMs: 2000,
    now: () => now,
    fetchImpl: async () => ({ ok: false, async json() { return {}; } }),
  });

  const payload = { target: "C", predicted: "O", errors: ["thumb_too_far"] };
  const first = await client.requestAdvice(payload);
  now = 500;
  const second = await client.requestAdvice({ ...payload, predicted: "Q" });
  assert.equal(first.advice.length > 0, true);
  assert.equal(second.source, "throttle");
});
