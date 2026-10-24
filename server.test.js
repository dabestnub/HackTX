const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const http = require("node:http");
const path = require("node:path");
const { once } = require("node:events");
const test = require("node:test");
const WebSocket = require("ws");

async function getFreePort() {
  const server = http.createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  return port;
}

async function startServer(port) {
  const child = spawn(process.execPath, [path.join(__dirname, "server.js")], {
    cwd: __dirname,
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => { stderr += chunk; });

  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Server exited early: ${stderr}`);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/`);
      if (response.ok) return child;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  child.kill();
  throw new Error(`Server did not start: ${stderr}`);
}

async function connect(port) {
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  const queue = [];
  const waiters = [];
  socket.on("message", (data) => {
    const message = JSON.parse(data.toString());
    const waiterIndex = waiters.findIndex((waiter) => waiter.predicate(message));
    if (waiterIndex >= 0) {
      const [waiter] = waiters.splice(waiterIndex, 1);
      clearTimeout(waiter.timeout);
      waiter.resolve(message);
    } else {
      queue.push(message);
    }
  });
  await once(socket, "open");
  socket.next = (predicate, timeoutMs = 2000) => {
    const queuedIndex = queue.findIndex(predicate);
    if (queuedIndex >= 0) return Promise.resolve(queue.splice(queuedIndex, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = {
        predicate,
        resolve,
        timeout: setTimeout(() => {
          const index = waiters.indexOf(waiter);
          if (index >= 0) waiters.splice(index, 1);
          reject(new Error(`Timed out waiting for a WebSocket message. Pending messages: ${JSON.stringify(queue)}`));
        }, timeoutMs),
      };
      waiters.push(waiter);
    });
  };
  return socket;
}

function send(socket, message) {
  socket.send(JSON.stringify(message));
}

test("two players share authoritative turns, words, and team score", async (t) => {
  const port = await getFreePort();
  const server = await startServer(port);
  const clients = [];
  t.after(() => {
    for (const client of clients) client.terminate();
    server.kill();
  });
  assert.equal((await fetch(`http://127.0.0.1:${port}/`)).status, 200);
  assert.equal((await fetch(`http://127.0.0.1:${port}/package.json`)).status, 404);

  const creator = await connect(port);
  clients.push(creator);
  const created = creator.next((message) => message.type === "room:joined");
  send(creator, { type: "room:create", name: "Alex" });
  const room = await created;
  assert.match(room.room.code, /^[A-HJ-NP-Z2-9]{6}$/);

  const partner = await connect(port);
  clients.push(partner);
  const joined = partner.next((message) => message.type === "room:joined");
  send(partner, { type: "room:join", code: room.room.code, name: "Sam" });
  await joined;
  const readySignals = Promise.all([
    creator.next((message) => message.type === "video:peer-ready"),
    partner.next((message) => message.type === "video:peer-ready"),
  ]);
  send(creator, { type: "video:ready" });
  send(partner, { type: "video:ready" });
  const peerReady = await readySignals;
  assert.deepEqual(peerReady.map((message) => message.peerSlot).sort(), [0, 1]);
  const relayedOffer = partner.next((message) => message.type === "video:signal");
  send(creator, {
    type: "video:signal",
    signal: { type: "offer", sdp: "test-session-description" },
  });
  const videoOffer = await relayedOffer;
  assert.equal(videoOffer.fromSlot, 0);
  assert.deepEqual(videoOffer.signal, { type: "offer", sdp: "test-session-description" });
  const peerStoppedSharing = partner.next((message) => message.type === "video:peer-left");
  send(creator, { type: "video:stop" });
  assert.equal((await peerStoppedSharing).peerSlot, 0);
  const started = Promise.all([
    creator.next((message) => message.type === "game:started"),
    partner.next((message) => message.type === "game:started"),
  ]);
  await started;

  const pausedRoom = partner.next((message) => message.type === "room:updated"
    && message.room.status === "waiting"
    && !message.room.players[0]?.connected);
  send(creator, { type: "game:leave" });
  await pausedRoom;

  const reconnected = await connect(port);
  clients.push(reconnected);
  const resumed = Promise.all([
    partner.next((message) => message.type === "game:started"),
    reconnected.next((message) => message.type === "game:started"),
  ]);
  const restored = reconnected.next((message) => message.type === "room:joined");
  send(reconnected, {
    type: "room:join",
    code: room.room.code,
    name: "Alex",
    token: room.player.token,
  });
  await restored;
  await resumed;

  const invalidTurn = partner.next((message) => message.type === "error");
  send(partner, { type: "letter:confirm", actionId: "0000000000000001", letter: "C" });
  assert.match((await invalidTurn).message, /other player's turn/i);

  const firstAccepted = Promise.all([
    reconnected.next((message) => message.type === "letter:accepted"),
    partner.next((message) => message.type === "letter:accepted"),
  ]);
  send(reconnected, { type: "letter:confirm", actionId: "0000000000000002", letter: "C" });
  await firstAccepted;

  const duplicateUpdate = reconnected.next((message) => message.type === "room:updated" && message.room.letters === "C");
  send(reconnected, { type: "letter:confirm", actionId: "0000000000000002", letter: "C" });
  assert.equal((await duplicateUpdate).room.letters, "C");

  let currentSlot = 1;
  let actionNumber = 3;
  const words = ["CAT", "DOG", "SUN", "HAT", "HOME"];
  for (const word of words) {
    for (const letter of word.slice(word === "CAT" ? 1 : 0)) {
      const actor = currentSlot === 0 ? reconnected : partner;
      const actionId = actionNumber.toString(16).padStart(16, "0");
      actionNumber += 1;
      if (word === "HOME" && letter === "E") {
        const completed = Promise.all([
          reconnected.next((message) => message.type === "game:completed"),
          partner.next((message) => message.type === "game:completed"),
        ]);
        send(actor, { type: "letter:confirm", actionId, letter });
        const results = await completed;
        for (const result of results) {
          assert.equal(result.room.score, 500);
          assert.equal(result.room.completedReason, "score");
        }
      } else {
        const accepted = Promise.all([
          reconnected.next((message) => message.type === "letter:accepted"),
          partner.next((message) => message.type === "letter:accepted"),
        ]);
        send(actor, { type: "letter:confirm", actionId, letter });
        await accepted;
      }
      currentSlot = 1 - currentSlot;
    }
  }
});
