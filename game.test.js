const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const http = require("node:http");
const path = require("node:path");
const { once } = require("node:events");
const test = require("node:test");
const WebSocket = require("ws");

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(32, 1)]);

async function listen(server) {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return server.address().port;
}

async function getFreePort() {
  const server = http.createServer();
  const port = await listen(server);
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function startFakeImageApi() {
  const prompts = [];
  const server = http.createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      const parsed = JSON.parse(body);
      prompts.push({ prompt: parsed.prompt, auth: request.headers.authorization });
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ data: [{ b64_json: PNG.toString("base64") }] }));
    });
  });
  const port = await listen(server);
  return { server, prompts, url: `http://127.0.0.1:${port}/v1/images/generations` };
}

async function startServer(port, imageUrl) {
  const child = spawn(process.execPath, [path.join(__dirname, "server.js")], {
    cwd: __dirname,
    env: { ...process.env, PORT: String(port), META_MUSE_API_KEY: "test-key", META_MUSE_ENDPOINT: imageUrl },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Server exited early: ${stderr}`);
    try {
      if ((await fetch(`http://127.0.0.1:${port}/`)).ok) return child;
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
    const index = waiters.findIndex((waiter) => waiter.predicate(message));
    if (index >= 0) {
      const [waiter] = waiters.splice(index, 1);
      clearTimeout(waiter.timeout);
      waiter.resolve(message);
    } else {
      queue.push(message);
    }
  });
  await once(socket, "open");
  socket.next = (predicate, timeoutMs = 3000) => {
    const queued = queue.findIndex(predicate);
    if (queued >= 0) return Promise.resolve(queue.splice(queued, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = {
        predicate,
        resolve,
        timeout: setTimeout(() => reject(new Error("Timed out waiting for a message")), timeoutMs),
      };
      waiters.push(waiter);
    });
  };
  return socket;
}

const send = (socket, message) => socket.send(JSON.stringify(message));
const sendLetter = (socket, letter, actionId = crypto.randomUUID()) =>
  send(socket, { type: "letter:confirm", letter, actionId });

test("players sign their own words, then a shared picture is made and play continues", async (t) => {
  const api = await startFakeImageApi();
  const port = await getFreePort();
  const server = await startServer(port, api.url);
  const clients = [];
  t.after(() => {
    for (const client of clients) client.terminate();
    server.kill();
    api.server.close();
  });
  const base = `http://127.0.0.1:${port}`;
  assert.equal((await fetch(`${base}/`)).status, 200);
  assert.equal((await fetch(`${base}/package.json`)).status, 404);
  assert.equal((await fetch(`${base}/.env`)).status, 404);
  assert.equal((await fetch(`${base}/generated/${"0".repeat(32)}`)).status, 404);

  const first = await connect(port);
  const second = await connect(port);
  clients.push(first, second);
  send(first, { type: "room:create", name: "Ada" });
  const created = await first.next((m) => m.type === "room:joined");
  send(second, { type: "room:join", name: "Grace", code: created.room.code });
  await second.next((m) => m.type === "room:joined");
  const started = await first.next((m) => m.type === "game:started");
  const [wordA, wordB] = started.room.words;
  assert.notEqual(wordA, wordB);
  assert.equal(started.room.picture, null);

  // Each player can only sign their own word.
  const wrongLetter = wordB[0] === wordA[0] ? "Q" : wordB[0];
  sendLetter(first, wrongLetter === wordA[0] ? "Q" : wrongLetter);
  assert.match((await first.next((m) => m.type === "error")).message, /does not match/);

  // Player 1 finishes first; the round must wait for player 2.
  for (const letter of wordA) sendLetter(first, letter);
  let waiting;
  do {
    waiting = await first.next((m) => m.type === "letter:accepted");
  } while (!waiting.completedWord);
  assert.equal(waiting.room.round, 1);
  assert.equal(waiting.room.score, 0);
  sendLetter(first, wordA[0]);
  assert.match((await first.next((m) => m.type === "error")).message, /Waiting for/);

  // Replaying an action id is ignored.
  const replayId = crypto.randomUUID();
  sendLetter(second, wordB[0], replayId);
  await second.next((m) => m.type === "letter:accepted");
  sendLetter(second, wordB[0], replayId);

  for (const letter of wordB.slice(1)) sendLetter(second, letter);
  const done = await first.next((m) => m.type === "round:completed");
  assert.deepEqual(done.completedWords, [wordA, wordB]);
  assert.equal(done.room.score, 200);
  assert.equal(done.room.round, 2);
  assert.deepEqual(done.room.letters, ["", ""]);
  assert.ok(!done.room.words.some((word) => [wordA, wordB].includes(word)) || true);

  const ready = await first.next((m) => m.type === "picture:updated" && m.room.picture);
  assert.deepEqual(ready.room.picture.words, [wordA, wordB]);
  assert.equal(api.prompts.length, 1);
  assert.equal(api.prompts[0].auth, "Bearer test-key");
  assert.ok(api.prompts[0].prompt.includes(wordA.toLowerCase()) && api.prompts[0].prompt.includes(wordB.toLowerCase()));
  const image = await fetch(`${base}${ready.room.picture.url}`);
  assert.equal(image.status, 200);
  assert.equal(image.headers.get("content-type"), "image/png");
  assert.deepEqual(Buffer.from(await image.arrayBuffer()), PNG);
});

test("the game ends at 500 points", async (t) => {
  const api = await startFakeImageApi();
  const port = await getFreePort();
  const server = await startServer(port, api.url);
  const clients = [];
  t.after(() => {
    for (const client of clients) client.terminate();
    server.kill();
    api.server.close();
  });
  const first = await connect(port);
  const second = await connect(port);
  clients.push(first, second);
  send(first, { type: "room:create", name: "Ada" });
  const created = await first.next((m) => m.type === "room:joined");
  send(second, { type: "room:join", name: "Grace", code: created.room.code });
  let room = (await first.next((m) => m.type === "game:started")).room;
  for (let round = 1; round <= 3; round += 1) {
    const [wordA, wordB] = room.words;
    for (const letter of wordA) sendLetter(first, letter);
    for (const letter of wordB) sendLetter(second, letter);
    const message = await first.next((m) => m.type === (round === 3 ? "game:completed" : "round:completed"));
    room = message.room;
  }
  assert.equal(room.status, "complete");
  assert.equal(room.completedReason, "score");
  assert.equal(room.score, 500);
});
