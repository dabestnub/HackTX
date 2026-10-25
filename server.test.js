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

  const initialWords = new Set();
  for (let i = 0; i < 12; i += 1) {
    const client = await connect(port);
    clients.push(client);
    send(client, { type: "room:create", name: "Word test" });
    const created = await client.next((message) => message.type === "room:joined");
    assert.match(created.room.target, /^[A-Z]{3,5}$/);
    initialWords.add(created.room.target);
    send(client, { type: "game:leave" });
  }
  assert.ok(initialWords.size > 1, "new rooms should not always start with the same word");

  const creator = await connect(port);
  clients.push(creator);
  const created = creator.next((message) => message.type === "room:joined");
  send(creator, { type: "room:create", name: "Alex", targetScore: 750 });
  const room = await created;
  assert.equal(room.room.targetScore, 750);
  assert.match(room.room.code, /^[A-HJ-NP-Z2-9]{6}$/);

  const partner = await connect(port);
  clients.push(partner);
  const joined = partner.next((message) => message.type === "room:joined");
  send(partner, { type: "room:join", code: room.room.code, name: "Sam" });
  const partnerRoom = await joined;
  assert.equal(partnerRoom.room.target, room.room.target);
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
  const restoredRoom = await restored;
  assert.equal(restoredRoom.room.target, room.room.target);
  await resumed;

  const invalidTurn = partner.next((message) => message.type === "error");
  send(partner, { type: "letter:confirm", actionId: "0000000000000001", letter: "C" });
  assert.match((await invalidTurn).message, /other player's turn/i);

  const wrongLetter = room.room.target[0] === "A" ? "B" : "A";
  send(reconnected, { type: "letter:confirm", actionId: "0000000000000099", letter: wrongLetter });
  assert.match((await reconnected.next((message) => message.type === "error")).message, /does not match/i);
  const firstLetter = room.room.target[0];
  const firstAccepted = Promise.all([
    reconnected.next((message) => message.type === "letter:accepted"),
    partner.next((message) => message.type === "letter:accepted"),
  ]);
  send(reconnected, { type: "letter:confirm", actionId: "0000000000000002", letter: firstLetter });
  await firstAccepted;

  const duplicateUpdate = reconnected.next((message) => message.type === "room:updated" && message.room.letters === firstLetter);
  send(reconnected, { type: "letter:confirm", actionId: "0000000000000002", letter: firstLetter });
  assert.equal((await duplicateUpdate).room.letters, firstLetter);

  let state = (await firstAccepted)[0].room;
  let actionNumber = 3;
  while (state.status !== "complete") {
    const word = state.target;
    const actor = state.currentPlayer === 0 ? reconnected : partner;
    const letter = word[state.letters.length];
    const completesWord = state.letters.length === word.length - 1;
    const type = completesWord && state.score + word.length * 25 >= state.targetScore ? "game:completed" : "letter:accepted";
    const updates = Promise.all([
      reconnected.next((message) => message.type === type),
      partner.next((message) => message.type === type),
    ]);
    send(actor, { type: "letter:confirm", actionId: (actionNumber++).toString(16).padStart(16, "0"), letter });
    const [first, second] = await updates;
    assert.deepEqual(first.room, second.room);
    assert.equal(first.room.score, state.score + (completesWord ? word.length * 25 : 0));
    if (completesWord) {
      assert.notEqual(first.room.target, word, "consecutive words must differ");
      assert.equal(first.room.letters, "");
    } else {
      assert.equal(first.room.target, word);
      assert.equal(first.room.letters, state.letters + letter);
    }
    state = first.room;
  }
  assert.ok(state.score >= 750);
  assert.ok(state.wordsCompleted >= 6);
  assert.equal(state.completedReason, "score");
  send(reconnected, {type: "game:restart", targetScore: -1});
  assert.match((await reconnected.next(m => m.type === "error")).message, /points/i);
  const restarted = Promise.all([
    reconnected.next(m => m.type === "game:started"),
    partner.next(m => m.type === "game:started"),
  ]);
  send(reconnected, {type: "game:restart", targetScore: 1000});
  for (const message of await restarted) {
    assert.equal(message.room.code, room.room.code);
    assert.equal(message.room.targetScore, 1000);
    assert.equal(message.room.score, 0);
    assert.equal(message.room.wordsCompleted, 0);
    assert.equal(message.room.letters, "");
    assert.equal(message.room.secondsRemaining, 120);
  }
  send(partner, {type: "game:restart", targetScore: 500});
  assert.match((await partner.next(m => m.type === "error")).message, /ended/i);
});
