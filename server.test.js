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

test("two players complete separate words, share video, and confirm a rematch", async (t) => {
  const port = await getFreePort();
  const server = await startServer(port);
  const clients = [];
  t.after(() => {
    for (const client of clients) client.terminate();
    server.kill();
  });
  assert.equal((await fetch(`http://127.0.0.1:${port}/`)).status, 200);
  const alphabetGuide = await fetch(`http://127.0.0.1:${port}/signLanguage.jpg`);
  assert.equal(alphabetGuide.status, 200);
  assert.equal(alphabetGuide.headers.get("content-type"), "image/jpeg");
  assert.equal((await fetch(`http://127.0.0.1:${port}/package.json`)).status, 404);

  const initialWordPairs = new Set();
  for (let i = 0; i < 12; i += 1) {
    const client = await connect(port);
    clients.push(client);
    send(client, { type: "room:create", name: "Word test" });
    const created = await client.next((message) => message.type === "room:joined");
    assert.equal(created.room.words.length, 2);
    assert.notEqual(created.room.words[0], created.room.words[1]);
    assert.equal(created.room.words[0].length, created.room.words[1].length);
    initialWordPairs.add(created.room.words.join("/"));
    send(client, { type: "game:leave" });
  }
  assert.ok(initialWordPairs.size > 1, "new rooms should not always start with the same words");

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
  assert.deepEqual(partnerRoom.room.words, room.room.words);
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
  send(creator, { type: "asl:ready" });
  send(partner, { type: "asl:ready" });
  let started = Promise.all([
    creator.next((message) => message.type === "game:started"),
    partner.next((message) => message.type === "game:started"),
  ]);
  let startedRooms = await started;
  assert.deepEqual(startedRooms.map((message) => message.room.status), ["playing", "playing"]);

  const pausedRoom = partner.next((message) => message.type === "room:updated"
    && message.room.status === "waiting"
    && !message.room.players[0]?.connected);
  send(creator, { type: "game:leave" });
  await pausedRoom;

  const reconnected = await connect(port);
  clients.push(reconnected);
  const restored = reconnected.next((message) => message.type === "room:joined");
  send(reconnected, {
    type: "room:join",
    code: room.room.code,
    name: "Alex",
    token: room.player.token,
  });
  const restoredRoom = await restored;
  assert.deepEqual(restoredRoom.room.words, room.room.words);
  const reconnectedPeers = Promise.all([
    partner.next((message) => message.type === "video:peer-ready"),
    reconnected.next((message) => message.type === "video:peer-ready"),
  ]);
  send(reconnected, { type: "video:ready" });
  send(partner, { type: "video:ready" });
  await reconnectedPeers;
  started = Promise.all([
    partner.next((message) => message.type === "game:started"),
    reconnected.next((message) => message.type === "game:started"),
  ]);
  send(reconnected, { type: "asl:ready" });
  startedRooms = await started;
  let state = startedRooms[0].room;
  let actionNumber = 1;
  let firstWordChecked = false;
  while (state.status !== "complete") {
    const words = state.words;
    roundLetters: for (let index = 0; index < words[0].length && state.status !== "complete"; index += 1) {
      for (const slot of [0, 1]) {
        if (state.status === "complete") break;
        const letter = words[slot][state.letters[slot].length];
        if (!firstWordChecked && slot === 0 && index === 0) {
          const wrongLetter = words[1][0] === "A" ? "B" : "A";
          send(partner, { type: "letter:confirm", actionId: "00000000000000ff", letter: wrongLetter });
          assert.match((await partner.next((message) => message.type === "error")).message, /does not match/i);
          firstWordChecked = true;
        }
        const nextLetters = state.letters.slice();
        nextLetters[slot] += letter;
        const completesRound = nextLetters.every((letters, i) => letters === words[i]);
        const completedScore = state.score + words.reduce((total, word) => total + word.length * 25, 0);
        const expectedType = completesRound
          ? (completedScore >= state.targetScore ? "game:completed" : "round:completed")
          : "letter:accepted";
        const updates = Promise.all([
          reconnected.next((message) => message.type === expectedType),
          partner.next((message) => message.type === expectedType),
        ]);
        const actor = slot === 0 ? reconnected : partner;
        send(actor, { type: "letter:confirm", actionId: (actionNumber++).toString(16).padStart(16, "0"), letter });
        const [first, second] = await updates;
        assert.deepEqual(first.room, second.room);
        if (completesRound) {
          assert.equal(first.room.score, completedScore);
          if (expectedType === "round:completed") {
            assert.notDeepEqual(first.room.words, words, "consecutive rounds must use different words");
            assert.deepEqual(first.room.letters, ["", ""]);
          } else {
            assert.deepEqual(first.room.letters, words);
          }
        } else {
          assert.equal(first.room.score, state.score);
          assert.equal(first.room.letters[slot], nextLetters[slot]);
        }
        state = first.room;
        if (completesRound) break roundLetters;
      }
    }
  }
  assert.ok(state.score >= 750);
  assert.ok(state.wordsCompleted >= 6);
  assert.equal(state.completedReason, "score");
  send(reconnected, {type: "game:restart", targetScore: -1});
  assert.match((await reconnected.next(m => m.type === "error")).message, /points/i);
  const waitingForRematch = partner.next((message) => message.type === "room:updated"
    && message.room.pendingRematch?.targetScore === 1000);
  send(reconnected, {type: "game:restart", targetScore: 1000, roundDuration: 180});
  const rematchPending = await waitingForRematch;
  assert.equal(rematchPending.room.players[0].rematchReady, true);
  assert.equal(rematchPending.room.players[1].rematchReady, false);
  const restarted = Promise.all([
    reconnected.next(m => m.type === "game:started"),
    partner.next(m => m.type === "game:started"),
  ]);
  send(partner, {type: "game:restart", targetScore: 500, roundDuration: 60});
  for (const message of await restarted) {
    assert.equal(message.room.code, room.room.code);
    assert.equal(message.room.targetScore, 1000);
    assert.equal(message.room.roundDuration, 180);
    assert.equal(message.room.score, 0);
    assert.equal(message.room.wordsCompleted, 0);
    assert.deepEqual(message.room.letters, ["", ""]);
    assert.equal(message.room.secondsRemaining, 180);
  }
  send(partner, {type: "game:restart", targetScore: 500});
  assert.match((await partner.next(m => m.type === "error")).message, /ended/i);
});
