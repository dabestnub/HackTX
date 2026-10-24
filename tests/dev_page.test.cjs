// Execute the shipped inline script; browser I/O is controlled at its boundaries.
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const html = readFileSync(join(__dirname, '../server/static/dev.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function browser(camera = async () => ({ getTracks: () => [] })) {
  let now = 0, nextTimer = 0;
  const timers = new Map(), sockets = [], blobs = [], draws = [];
  const ctx = { clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {},
    drawImage: (...args) => draws.push(args) };
  const element = () => ({ textContent: '', innerHTML: '', value: '', disabled: false, hidden: false,
    style: {}, width: 640, height: 480, firstElementChild: { style: {} },
    setAttribute(name, value) { this[name] = value; },
    replaceChildren(...children) { this.children = children; this.textContent = ''; },
    getContext: () => ctx, toBlob: (callback, type) => blobs.push({ callback, type }),
    play: async () => {} });
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map((m) => [m[1], element()]));
  elements.name.value = 'Player';
  elements.code.value = 'ABCD';
  elements.video.videoWidth = 1280;
  elements.video.videoHeight = 720;
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 0; this.bufferedAmount = 0; this.sent = []; sockets.push(this); }
    open() { this.readyState = 1; this.onopen?.(); }
    message(message) { this.onmessage?.({ data: JSON.stringify(message) }); }
    send(message) { assert.equal(this.readyState, 1); this.sent.push(message); }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  const timeout = (fn, delay, interval = false) => {
    const id = ++nextTimer;
    timers.set(id, { fn, at: now + delay, delay, interval });
    return id;
  };
  const context = vm.createContext({ document: { getElementById: (id) => elements[id], createElement: element },
    location: { protocol: 'http:', host: 'localhost' }, navigator: { mediaDevices: { getUserMedia: camera } },
    WebSocket: Socket, Image: class {}, performance: { now: () => now },
    setTimeout: (fn, delay) => timeout(fn, delay), clearTimeout: (id) => timers.delete(id),
    setInterval: (fn, delay) => timeout(fn, delay, true), clearInterval: (id) => timers.delete(id),
    addEventListener() {}, console });
  vm.runInContext(script, context);
  function advance(ms) {
    const end = now + ms;
    for (;;) {
      const entry = [...timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!entry) break;
      const [id, timer] = entry;
      now = timer.at;
      if (timer.interval) timer.at += timer.delay;
      else timers.delete(id);
      timer.fn();
    }
    now = end;
  }
  function state(socket, overrides = {}) {
    socket.message({ type: 'room_state', room: 'ABCD', phase: 'signing', paused: false,
      score: 0, target: 500, round: 1, time_left: 10,
      players: [{ slot: 1, name: 'Player', ready: true, connected: true },
        { slot: 2, name: 'Friend', ready: true, connected: true }], ...overrides });
  }
  async function joined() {
    await elements.create.onclick();
    const socket = sockets.at(-1);
    socket.open();
    socket.message({ type: 'room_joined', room: 'ABCD', slot: 1, token: 'secret' });
    state(socket);
    return socket;
  }
  return { elements, sockets, blobs, draws, advance, state, joined, context };
}

test('camera denial is actionable and never creates a socket', async () => {
  const b = browser(async () => { throw Object.assign(new Error('Denied'), { name: 'NotAllowedError' }); });
  await assert.doesNotReject(() => b.elements.create.onclick());
  assert.match(b.elements.status.textContent, /camera.*(permission|allow|denied)/i);
  assert.equal(b.sockets.length, 0);
  assert.equal(b.elements.create.disabled, false);
});

test('unavailable camera reports a secure-origin remedy', async () => {
  const b = browser();
  vm.runInContext('navigator.mediaDevices = undefined', b.context);
  await assert.doesNotReject(() => b.elements.create.onclick());
  assert.match(b.elements.status.textContent, /HTTPS|localhost/);
  assert.equal(b.sockets.length, 0);
});

test('pending camera request and connected room reject duplicate connection clicks', async () => {
  let allow;
  const permission = new Promise((resolve) => { allow = resolve; });
  const b = browser(() => permission);
  const first = b.elements.create.onclick();
  const duplicate = b.elements.join.onclick();
  allow({ getTracks: () => [] });
  await Promise.all([first, duplicate]);
  assert.equal(b.sockets.length, 1);
  b.sockets[0].open();
  b.sockets[0].message({ type: 'room_joined', room: 'ABCD', slot: 1, token: 'secret' });
  await b.elements.create.onclick();
  assert.equal(b.sockets.length, 1);
});

test('reconnect stops within server grace even when a retry socket hangs', async () => {
  const b = browser(), socket = await b.joined();
  socket.close();
  b.advance(2000);
  assert.equal(b.sockets.length, 2);
  const retry = b.sockets[1];
  retry.open();
  assert.deepEqual(JSON.parse(retry.sent[0]), { type: 'rejoin', room: 'ABCD', token: 'secret' });
  retry.close();
  b.advance(28000);
  assert.ok(b.sockets.every((s) => s.readyState === 3));
  const count = b.sockets.length;
  b.advance(60000);
  assert.equal(b.sockets.length, count);
  assert.match(b.elements.status.textContent, /expired|ended/i);
  assert.equal(b.elements.ready.disabled, true);
});

test('existing partner pause bounds reconnect to the original room grace', async () => {
  const b = browser(), socket = await b.joined();
  b.state(socket, { paused: true, time_left: 8 });
  b.advance(20000);
  socket.close();
  b.advance(10000);
  assert.ok(b.sockets.every((s) => s.readyState === 3));
});

test('terminal server errors and room closure stop retries and remain visible', async () => {
  for (const message of [{ type: 'error', code: 'recognizer_unavailable', message: 'Camera recognition is unavailable.' },
    { type: 'error', code: 'bad_token', message: 'Invalid token.' },
    { type: 'room_closed', reason: 'partner_left' }]) {
    const b = browser(), socket = await b.joined();
    socket.message(message);
    socket.close();
    b.advance(60000);
    assert.equal(b.sockets.length, 1);
    assert.match(b.elements.status.textContent, /unavailable|token|partner_left/);
    assert.equal(b.elements.ready.disabled, true);
  }
});

test('JPEG capture preserves raw aspect ratio, one encoder, 10fps, and backpressure', async () => {
  const b = browser(), socket = await b.joined();
  b.elements.video.onloadedmetadata();
  assert.equal(b.elements.stage.style.aspectRatio, '1280 / 720');
  b.advance(100);
  assert.equal(b.blobs.length, 1);
  assert.equal(b.blobs[0].type, 'image/jpeg');
  assert.deepEqual(b.draws[0].slice(1), [0, 0, 640, 360]);
  b.advance(99);
  assert.equal(b.blobs.length, 1);
  const jpeg = { size: 1000 };
  b.blobs[0].callback(jpeg);
  b.advance(1);
  assert.equal(socket.sent.filter((v) => v === jpeg).length, 1);
  if (b.blobs[1]) b.blobs[1].callback(jpeg);
  assert.equal(socket.sent.filter((v) => v === jpeg).length, 1, 'no burst after a delayed encode');
  socket.bufferedAmount = 1000;
  const count = b.blobs.length;
  b.advance(500);
  assert.equal(b.blobs.length, count);
});

test('pending JPEG is dropped after signing ends, even if a new signing phase starts', async () => {
  const b = browser(), socket = await b.joined();
  b.advance(100);
  b.state(socket, { phase: 'generating', time_left: null });
  b.state(socket, { round: 2 });
  const jpeg = { size: 1000 };
  b.blobs[0].callback(jpeg);
  assert.equal(socket.sent.includes(jpeg), false);
});

test('pending JPEG cannot send on replacement socket after rejoin', async () => {
  const b = browser(), socket = await b.joined();
  b.advance(100);
  socket.close();
  b.advance(2000);
  const retry = b.sockets[1];
  retry.open();
  retry.message({ type: 'room_joined', room: 'ABCD', slot: 1, token: 'secret' });
  b.state(retry);
  const jpeg = { size: 1000 };
  b.blobs[0].callback(jpeg);
  assert.equal(retry.sent.includes(jpeg), false);
});

test('timer freezes during pause and resumes from server time_left', async () => {
  const b = browser(), socket = await b.joined();
  b.advance(2000);
  b.state(socket, { paused: true, time_left: 8 });
  assert.equal(b.elements.timer.textContent, '8.0s');
  b.advance(5000);
  assert.equal(b.elements.timer.textContent, '8.0s');
  b.state(socket, { time_left: 8 });
  b.advance(1000);
  assert.equal(b.elements.timer.textContent, '7.0s');
});

test('room state preserves countdown notice until signing begins', async () => {
  const b = browser(), socket = await b.joined();
  socket.message({ type: 'countdown', round: 2, seconds: 3 });
  b.state(socket, { phase: 'countdown', time_left: null, round: 2 });
  assert.match(b.elements.timer.textContent, /starting in 3s/);
});

test('ready, word progress, image, gallery, and remote play-again reset follow server state', async () => {
  const b = browser(), socket = await b.joined();
  b.state(socket, { phase: 'lobby', time_left: null });
  assert.equal(b.elements.ready.disabled, false);
  b.elements.ready.onclick();
  assert.deepEqual(JSON.parse(socket.sent.at(-1)), { type: 'ready', ready: false });
  socket.message({ type: 'round_start', round: 1, words: [{ slot: 1, word: 'RED' }, { slot: 2, word: 'CAT' }] });
  socket.message({ type: 'progress', slot: 1, done: 1, length: 3 });
  assert.match(b.elements.mine.innerHTML, /class="done">R/);
  socket.message({ type: 'image_ready', url: '/images/ABCD/1', phrase: 'RED CAT' });
  assert.equal(b.elements.image.children[0].alt, 'RED CAT');
  socket.message({ type: 'game_over', score: 500, gallery: [{ url: '/images/ABCD/1', phrase: 'RED CAT' }] });
  b.state(socket, { phase: 'game_over', time_left: null, score: 500 });
  assert.equal(b.elements.again.hidden, false);
  assert.equal(b.elements.gallery.children.length, 1);
  b.state(socket, { phase: 'lobby', time_left: null, score: 0 });
  assert.equal(b.elements.gallery.children.length, 0);
  assert.equal(b.elements.mine.innerHTML, '');
});
