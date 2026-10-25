const assert = require('node:assert/strict');
const test = require('node:test');
const { HybridRecognizer, normalizeLandmarks, addVelocityFeatures } = require('./asl-recognition');
const config = require('./assets/asl_model_config.json');

function hand(change = 0) {
  return Array.from({ length: 21 }, (_, index) => ({
    x: 0.5 + index * 0.01 + (index === 8 ? change : 0),
    y: 0.6 - index * 0.02,
    z: index * 0.001,
  }));
}

function fixture() {
  const state = { staticLabel: 'A', motionLabel: 'J', lowConfidence: false, calls: [] };
  const ort = { Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } } };
  const session = kind => ({ async run({ landmarks }) {
    state.calls.push({ kind, landmarks });
    const classes = config[kind + '_classes'];
    const logits = classes.map(label => state.lowConfidence ? 0 : label === state[kind + 'Label'] ? 10 : 0);
    return { logits: { data: Float32Array.from(logits) } };
  } });
  const recognizer = new HybridRecognizer(ort, config, session('static'), session('motion'));
  return { recognizer, state };
}

// Captured landmarks are rounded to float32 before wrist/palm normalisation.
test('landmark normalization is translation/scale invariant and rejects degenerate input', () => {
  const input = hand();
  const output = normalizeLandmarks(input);
  assert.equal(output.length, 63);
  assert.deepEqual([...output.slice(0, 3)], [0, 0, 0]);
  assert.ok(Math.abs(Math.hypot(...output.slice(27, 30)) - 1) < 1e-6);
  const transformed = normalizeLandmarks(input.map(p => ({ x: p.x * 2 + 1, y: p.y * 2 - 1, z: p.z * 2 + 3 })));
  output.forEach((value, index) => assert.ok(Math.abs(value - transformed[index]) < 2e-6));
  assert.equal(normalizeLandmarks(Array(21).fill({ x: 0, y: 0, z: 0 })), null);
  assert.equal(normalizeLandmarks([]), null);
  assert.equal(normalizeLandmarks(hand().map(p => ({ ...p, z: NaN }))), null);
});

test('motion features use positions then velocity with a zero initial velocity', () => {
  const frames = [new Float32Array(63).fill(1), new Float32Array(63).fill(3)];
  const features = addVelocityFeatures(frames);
  assert.equal(features.length, 252);
  assert.ok(features.slice(0, 63).every(x => x === 1));
  assert.ok(features.slice(63, 126).every(x => x === 0));
  assert.ok(features.slice(126, 189).every(x => x === 3));
  assert.ok(features.slice(189).every(x => x === 2));
});

test('static inference uses checkpoint scaling and filters uncertain predictions', async () => {
  const { recognizer, state } = fixture();
  const result = await recognizer.predict(hand(), 0);
  assert.equal(result.label, 'A');
  assert.equal(result.mode, 'static');
  assert.ok(result.score > 0.99);
  const input = state.calls[0].landmarks;
  const norm = normalizeLandmarks(hand());
  norm.forEach((value, i) => assert.equal(input.data[i], Math.fround(Math.fround(value - config.scaler_mean[i]) / config.scaler_scale[i])));
  assert.deepEqual(input.dims, [1, 63]);
  state.lowConfidence = true;
  assert.equal((await recognizer.predict(hand(), 33)).label, '');
});

test('30-frame motion model distinguishes J/Z and rejects STATIC transitions', async () => {
  const { recognizer, state } = fixture();
  let result;
  for (let i = 0; i < 30; i++) result = await recognizer.predict(hand(i % 2 ? 0.3 : 0), i * 34);
  assert.equal(result.mode, 'motion');
  assert.equal(result.label, 'J');
  assert.deepEqual(state.calls.at(-1).landmarks.dims, [1, 30, 126]);
  state.motionLabel = 'Z';
  assert.equal((await recognizer.predict(hand(0.3), 30 * 34)).label, 'Z');
  state.motionLabel = 'STATIC';
  assert.equal((await recognizer.predict(hand(0), 31 * 34)).label, '');
  state.motionLabel = 'Z';
  state.lowConfidence = true;
  assert.equal((await recognizer.predict(hand(0.3), 32 * 34)).label, '');
  state.lowConfidence = false;
  for (let i = 33; i < 65; i++) result = await recognizer.predict(hand(), i * 34);
  assert.equal(result.mode, 'static');
});

test('missing hands, pauses and frame gaps clear motion and smoothing history', async () => {
  const { recognizer, state } = fixture();
  for (let i = 0; i < 30; i++) await recognizer.predict(hand(i % 2 ? 0.3 : 0), i * 34);
  await recognizer.predict(null, 30 * 34);
  assert.equal(recognizer.frames.length, 0);
  assert.equal(recognizer.motionActive, false);
  state.staticLabel = 'B';
  assert.equal((await recognizer.predict(hand(), 1100)).label, 'B');
  state.staticLabel = 'C';
  assert.equal((await recognizer.predict(hand(), 2000)).label, 'C');
  assert.equal(recognizer.frames.length, 1);
  recognizer.reset();
  assert.equal(recognizer.history.length, 0);
});

test('in-flight inference is discarded when the camera or prompt resets', async () => {
  const { recognizer } = fixture();
  let finish;
  recognizer.staticSession.run = () => new Promise(resolve => { finish = resolve; });
  const pending = recognizer.predict(hand(), 0);
  recognizer.reset();
  finish({ logits: { data: Float32Array.from(config.static_classes, x => x === 'A' ? 10 : 0) } });
  assert.equal((await pending).label, '');
  assert.equal(recognizer.history.length, 0);
});

test('incompatible model metadata fails before inference', () => {
  assert.throws(() => new HybridRecognizer({}, { ...config, motion_classes: ['J', 'Z'] }), /invalid/);
  assert.throws(() => new HybridRecognizer({}, { ...config, scaler_scale: Array(63).fill(0) }), /invalid/);
});

test('a recognized J survives its I-shaped ending for the full 1.5-second hold', () => {
  const { recognizer } = fixture();
  const j = { label: 'J', score: 0.95, mode: 'motion' };
  const i = { label: 'I', score: 0.92, mode: 'static' };
  assert.equal(recognizer.confirmPrediction(i, hand(), 0, 1500).label, 'I');
  for (let time = 100; time <= 900; time += 100) {
    assert.equal(recognizer.confirmPrediction(j, hand(), time, 1500).label, 'J');
  }
  for (let time = 1000; time <= 1600; time += 100) {
    const result = recognizer.confirmPrediction(i, hand(), time, 1500);
    assert.equal(result.label, 'J');
    assert.equal(result.holdingMotion, true);
  }
  // The original gesture cannot remain armed indefinitely.
  for (let time = 1700; time <= 2500; time += 100) {
    const result = recognizer.confirmPrediction(i, hand(), time, 1500);
    if (time === 2500) assert.equal(result.label, 'I');
  }
});

test('motion confirmation rejects isolated noise and cancels on a changed pose or sign', () => {
  const { recognizer } = fixture();
  const j = { label: 'J', score: 0.95, mode: 'motion' };
  const i = { label: 'I', score: 0.92, mode: 'static' };
  recognizer.confirmPrediction(j, hand(), 0, 1500);
  assert.equal(recognizer.confirmPrediction(i, hand(), 100, 1500).label, 'I');
  for (let time = 200; time <= 600; time += 100) recognizer.confirmPrediction(j, hand(), time, 1500);
  assert.equal(recognizer.confirmPrediction(i, hand(0.3), 700, 1500).label, 'I');
  for (let time = 800; time <= 1200; time += 100) recognizer.confirmPrediction(j, hand(), time, 1500);
  assert.equal(recognizer.confirmPrediction({ ...j, label: 'Z' }, hand(), 1300, 1500).label, 'Z');
  assert.equal(recognizer.confirmPrediction(i, hand(), 1400, 1500).label, 'I');
});

test('Z can finish confirmation in its stationary ending pose', () => {
  const { recognizer } = fixture();
  for (let time = 0; time <= 800; time += 100) {
    recognizer.confirmPrediction({ label: 'Z', score: 0.95, mode: 'motion' }, hand(), time, 1500);
  }
  for (let time = 900; time <= 1500; time += 100) {
    assert.equal(recognizer.confirmPrediction({ label: 'X', score: 0.9, mode: 'static' }, hand(), time, 1500).label, 'Z');
  }
});

test('motion confirmation resets on missing hands, frame gaps and explicit pauses', () => {
  const j = { label: 'J', score: 0.95, mode: 'motion' };
  const i = { label: 'I', score: 0.92, mode: 'static' };
  for (const interrupt of ['missing', 'gap', 'reset']) {
    const { recognizer } = fixture();
    for (let time = 0; time <= 800; time += 100) recognizer.confirmPrediction(j, hand(), time, 1500);
    if (interrupt === 'missing') recognizer.confirmPrediction({ label: '', score: 0, mode: 'static' }, null, 900, 1500);
    if (interrupt === 'reset') recognizer.reset();
    assert.equal(recognizer.confirmPrediction(i, hand(), interrupt === 'gap' ? 1400 : 1000, 1500).label, 'I');
  }
});
