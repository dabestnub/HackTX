const assert = require("node:assert/strict");
const test = require("node:test");
const { evaluateLandmarks } = require("./landmark-rules");

function makeLandmarks() {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  points[8] = { x: 0.52, y: 0.52, z: 0 };
  points[6] = { x: 0.53, y: 0.53, z: 0 };
  points[12] = { x: 0.53, y: 0.53, z: 0 };
  points[4] = { x: 0.49, y: 0.49, z: 0 };
  points[3] = { x: 0.50, y: 0.50, z: 0 };
  points[0] = { x: 0.5, y: 0.5, z: 0 };
  return points;
}

test("evaluateLandmarks returns pass when shape is stable and prediction matches", () => {
  const result = evaluateLandmarks({
    target: "C",
    predicted: "C",
    confidence: 0.93,
    holdMs: 1500,
    landmarks: makeLandmarks(),
  });
  assert.equal(result.pass, true);
  assert.equal(result.errors.length <= 1, true);
});

test("evaluateLandmarks flags missing hand", () => {
  const result = evaluateLandmarks({ target: "A", predicted: "A", confidence: 1, holdMs: 2000, landmarks: [] });
  assert.equal(result.pass, false);
  assert.deepEqual(result.errors, ["no_hand"]);
});
