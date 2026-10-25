const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { WebSocketServer, WebSocket } = require("ws");

const ROOT = __dirname;
try {
  if (typeof process.loadEnvFile === "function") process.loadEnvFile(path.join(ROOT, ".env"));
} catch (error) {
  if (error.code !== "ENOENT") console.warn(`Could not read .env: ${error.message}`);
}

const PORT = Number(process.env.PORT || 8000);
const ROUND_SECONDS = 120;
const POINTS_PER_LETTER = 25;
const WORDS_PER_ROUND = 2;
const WORDS = [
  "CAT", "DOG", "SUN", "HAT", "HOME", "LOVE", "BIRD", "FISH", "TREE", "BOOK",
  "MOON", "STAR", "RAIN", "WIND", "LEAF", "BEAR", "FROG", "DUCK", "BOAT", "SHIP",
  "CAKE", "MILK", "PEAR", "APPLE", "BREAD", "SMILE", "HAPPY", "LIGHT", "GREEN", "BLUE",
  "RED", "GOLD", "HAND", "WAVE", "PLAY", "SONG", "DREAM", "CLOUD", "WATER", "EARTH",
];
const IMAGE_ENDPOINT = process.env.META_MUSE_ENDPOINT || "https://api.meta.ai/v1/images/generations";
const IMAGE_MODEL = process.env.META_MUSE_MODEL || "muse-image-1.0";
const IMAGE_SIZE = process.env.META_MUSE_SIZE || "1024x1024";
const IMAGE_TIMEOUT_MS = Number(process.env.IMAGE_TIMEOUT_MS || 120_000);
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const MAX_IMAGES_PER_ROOM = 4;
const MAX_IMAGES_IN_FLIGHT = 3;
const MAX_IMAGES_PER_MINUTE = 20;
const GENERATED_PATH = /^\/generated\/([0-9a-f]{32})$/;
const PUBLIC_FILES = new Set([
  "/index.html",
  "/assets/asl_classes.json",
  "/assets/asl_classifier.onnx",
]);
const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".svg": "image/svg+xml",
};
const rooms = new Map();
const sockets = new Map();
const images = new Map();
let imageJobsInFlight = 0;
const imageStarts = [];

function randomCode() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from(crypto.randomBytes(6), (byte) => alphabet[byte % alphabet.length]).join("");
}

function makeRoom(targetScore, roundDuration = ROUND_SECONDS) {
  let code;
  do {
    code = randomCode();
  } while (rooms.has(code));
  const room = {
    code,
    players: [null, null],
    status: "waiting",
    round: 1,
    words: pickWords(),
    letters: ["", ""],
    score: 0,
    targetScore,
    roundDuration,
    wordsCompleted: 0,
    deadline: null,
    remainingMs: roundDuration * 1000,
    completedReason: null,
    pendingRematch: null,
    picture: null,
    pictureJob: null,
    imageIds: [],
    actionIds: new Set(),
    lastEmptyAt: null,
  };
  rooms.set(code, room);
  return room;
}

function pickWords(previous = []) {
  const excluded = new Set(previous);
  const byLength = new Map();
  for (const word of WORDS.filter((word) => !excluded.has(word))) {
    if (!byLength.has(word.length)) byLength.set(word.length, []);
    byLength.get(word.length).push(word);
  }
  let eligible = [...byLength.values()].filter((words) => words.length >= WORDS_PER_ROUND);
  if (!eligible.length) {
    const allByLength = new Map();
    for (const word of WORDS) {
      if (!allByLength.has(word.length)) allByLength.set(word.length, []);
      allByLength.get(word.length).push(word);
    }
    eligible = [...allByLength.values()].filter((words) => words.length >= WORDS_PER_ROUND);
  }
  const bucket = eligible[crypto.randomInt(eligible.length)];
  const first = bucket[crypto.randomInt(bucket.length)];
  const secondChoices = bucket.filter((word) => word !== first);
  return [first, secondChoices[crypto.randomInt(secondChoices.length)]];
}

function roomSnapshot(room) {
  const secondsRemaining = room.status === "playing"
    ? Math.max(0, Math.ceil((room.deadline - Date.now()) / 1000))
    : Math.ceil(room.remainingMs / 1000);
  return {
    code: room.code,
    status: room.status,
    players: room.players.map((player) => player && ({
      name: player.name,
      connected: player.connected,
      aslReady: player.aslReady,
      videoSharing: player.videoReady,
      rematchReady: player.rematchReady,
    })),
    pendingRematch: room.pendingRematch,
    round: room.round,
    words: room.words,
    letters: room.letters,
    score: room.score,
    targetScore: room.targetScore,
    roundDuration: room.roundDuration,
    wordsCompleted: room.wordsCompleted,
    secondsRemaining,
    completedReason: room.completedReason,
    picture: room.picture,
    pictureJob: room.pictureJob,
  };
}

function send(socket, payload) {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function broadcast(room, type, extra = {}) {
  const message = { type, room: roomSnapshot(room), ...extra };
  for (const player of room.players) {
    if (player?.socket) send(player.socket, message);
  }
}

function startRoom(room) {
  if (!room.players.every((player) => player?.connected)) return;
  if (room.status === "waiting") {
    room.status = "ready";
    broadcast(room, "room:updated");
  }
  if (room.status !== "ready" || !room.players.every((player) => player.aslReady && player.videoReady)) return;
  room.status = "playing";
  room.deadline = Date.now() + room.remainingMs;
  broadcast(room, "game:started");
}

function attachPlayer(socket, room, slot, name, reconnectToken = null) {
  const token = reconnectToken || crypto.randomBytes(32).toString("hex");
  const previous = room.players[slot];
  if (previous?.socket && previous.socket !== socket) {
    sockets.delete(previous.socket);
    previous.socket.close(4001, "Reconnected from another tab");
  }
  room.players[slot] = {
    id: previous?.id || crypto.randomUUID(),
    name,
    reconnectToken: token,
    connected: true,
    aslReady: false,
    videoReady: false,
    rematchReady: false,
    socket,
    slot,
  };
  room.lastEmptyAt = null;
  sockets.set(socket, { room, slot });
  send(socket, {
    type: "room:joined",
    room: roomSnapshot(room),
    player: { slot, token, name },
  });
  broadcast(room, "room:updated");
  startRoom(room);
}

function cleanName(value) {
  if (typeof value !== "string") return null;
  const name = value.trim().replace(/\s+/g, " ");
  if (!name || name.length > 24 || /[\u0000-\u001f\u007f]/.test(name)) return null;
  return name;
}

function fail(socket, message) {
  send(socket, { type: "error", message });
}

function validTargetScore(value) {
  return Number.isInteger(value) && value >= 100 && value <= 5000 && value % 25 === 0;
}

function validRoundDuration(value) {
  return [60, 120, 180, 300].includes(value);
}

function handleRestart(socket, message) {
  const membership = sockets.get(socket);
  if (!membership) return fail(socket, "Join a room before starting another round.");
  const { room } = membership;
  if (room.status !== "complete") return fail(socket, "Wait until the round has ended to play again.");
  const targetScore = message.targetScore ?? room.targetScore;
  if (!validTargetScore(targetScore)) return fail(socket, "Choose a points target from 100 to 5000 in steps of 25.");
  const roundDuration = message.roundDuration ?? room.roundDuration;
  if (!validRoundDuration(roundDuration)) return fail(socket, "Choose a round duration of 1, 2, 3, or 5 minutes.");
  const player = room.players[membership.slot];
  if (!room.pendingRematch) room.pendingRematch = { targetScore, roundDuration };
  player.rematchReady = true;
  if (!room.players.every((entry) => entry?.connected && entry.rematchReady)) {
    broadcast(room, "room:updated");
    return;
  }
  room.targetScore = room.pendingRematch.targetScore;
  room.roundDuration = room.pendingRematch.roundDuration;
  room.pendingRematch = null;
  for (const entry of room.players) entry.rematchReady = false;
  room.words = pickWords(room.words);
  room.letters = ["", ""];
  room.round += 1;
  room.status = "waiting";
  room.score = 0;
  room.wordsCompleted = 0;
  room.remainingMs = room.roundDuration * 1000;
  room.deadline = null;
  room.completedReason = null;
  // Keep prior action IDs so delayed submissions cannot count in the new round.
  broadcast(room, "room:updated");
  startRoom(room);
}

function handleCreate(socket, message) {
  if (sockets.has(socket)) return fail(socket, "Leave your current room before creating another.");
  const name = cleanName(message.name);
  if (!name) return fail(socket, "Enter a nickname up to 24 characters.");
  const targetScore = message.targetScore ?? 500;
  if (!validTargetScore(targetScore)) return fail(socket, "Choose a points target from 100 to 5000 in steps of 25.");
  const roundDuration = message.roundDuration ?? ROUND_SECONDS;
  if (!validRoundDuration(roundDuration)) return fail(socket, "Choose a round duration of 1, 2, 3, or 5 minutes.");
  attachPlayer(socket, makeRoom(targetScore, roundDuration), 0, name);
}

function handleJoin(socket, message) {
  if (sockets.has(socket)) return fail(socket, "Leave your current room before joining another.");
  const name = cleanName(message.name);
  const code = typeof message.code === "string" ? message.code.trim().toUpperCase() : "";
  if (!name) return fail(socket, "Enter a nickname up to 24 characters.");
  if (!/^[A-HJ-NP-Z2-9]{6}$/.test(code)) return fail(socket, "Room codes are six letters or numbers.");
  const room = rooms.get(code);
  if (!room) return fail(socket, "Room not found. Check the code or create a new room.");

  if (typeof message.token === "string") {
    const slot = room.players.findIndex((player) => player?.reconnectToken === message.token);
    if (slot >= 0) {
      attachPlayer(socket, room, slot, room.players[slot].name, message.token);
      return;
    }
  }
  if (room.status === "complete") return fail(socket, "This round has ended. Create a new room to play again.");
  const slot = room.players.findIndex((player) => !player || !player.connected);
  if (slot < 0) return fail(socket, "This room already has two connected players.");
  if (room.players[slot]) room.players[slot] = null;
  attachPlayer(socket, room, slot, name);
}

function finishRoom(room, reason) {
  if (room.status === "complete") return;
  if (room.status === "playing") {
    room.remainingMs = Math.max(0, room.deadline - Date.now());
  }
  room.status = "complete";
  room.completedReason = reason;
  room.deadline = null;
  broadcast(room, "game:completed");
}

function detachPlayer(socket) {
  const membership = sockets.get(socket);
  if (!membership) return;
  sockets.delete(socket);
  const { room, slot } = membership;
  const player = room.players[slot];
  const peer = room.players[1 - slot];
  if (player?.videoReady && peer?.socket) {
    send(peer.socket, { type: "video:peer-left", peerSlot: slot });
    peer.videoReady = false;
  }
  if (player) {
    player.connected = false;
    player.videoReady = false;
    player.socket = null;
  }
  if (room.status === "playing") {
    room.remainingMs = Math.max(0, room.deadline - Date.now());
    room.deadline = null;
    room.status = "waiting";
  } else if (room.status === "ready") {
    room.status = "waiting";
  }
  room.lastEmptyAt = room.players.every((entry) => !entry?.connected) ? Date.now() : null;
  broadcast(room, "room:updated");
}

class ImageError extends Error {}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
const JPEG_SIGNATURE = Buffer.from([0xff, 0xd8, 0xff]);

function sniffImageType(bytes) {
  if (bytes.subarray(0, 4).equals(PNG_SIGNATURE)) return "image/png";
  if (bytes.subarray(0, 3).equals(JPEG_SIGNATURE)) return "image/jpeg";
  if (bytes.subarray(0, 4).toString("latin1") === "RIFF" && bytes.subarray(8, 12).toString("latin1") === "WEBP") {
    return "image/webp";
  }
  return null;
}

function buildImagePrompt([first, second]) {
  const a = first.toLowerCase();
  const b = second.toLowerCase();
  return "Create one playful cartoon illustration of a single fused concept: "
    + `'${a} ${b}'. Clearly include elements of '${a}' and '${b}' within one cohesive scene. `
    + "Use cheerful colors and a simple background. No text, letters, captions, or watermarks.";
}

async function requestImage(words) {
  const apiKey = process.env.META_MUSE_API_KEY;
  if (!apiKey) throw new ImageError("Picture generation isn't set up on this server.");
  const response = await fetch(IMAGE_ENDPOINT, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: IMAGE_MODEL,
      prompt: buildImagePrompt(words),
      n: 1,
      size: IMAGE_SIZE,
      response_format: "b64_json",
      output_format: "png",
    }),
    signal: AbortSignal.timeout(IMAGE_TIMEOUT_MS),
  });
  if (!response.ok) {
    console.error(`Image API error ${response.status}: ${(await response.text()).slice(0, 300)}`);
    throw new ImageError("The picture service couldn't make this picture.");
  }
  const encoded = (await response.json())?.data?.[0]?.b64_json;
  if (typeof encoded !== "string" || !encoded) throw new ImageError("The picture service returned no image.");
  const bytes = Buffer.from(encoded, "base64");
  const type = sniffImageType(bytes);
  if (!type || bytes.length > MAX_IMAGE_BYTES) throw new ImageError("The picture service returned an unusable image.");
  return { bytes, type };
}

function storeImage(room, image) {
  const id = crypto.randomBytes(16).toString("hex");
  images.set(id, image);
  room.imageIds.push(id);
  while (room.imageIds.length > MAX_IMAGES_PER_ROOM) images.delete(room.imageIds.shift());
  return id;
}

function forgetImages(room) {
  for (const id of room.imageIds) images.delete(id);
  room.imageIds = [];
}

function reserveImageSlot() {
  const now = Date.now();
  while (imageStarts.length && now - imageStarts[0] > 60_000) imageStarts.shift();
  if (imageJobsInFlight >= MAX_IMAGES_IN_FLIGHT || imageStarts.length >= MAX_IMAGES_PER_MINUTE) {
    throw new ImageError("The picture service is busy. The next round's picture should work.");
  }
  imageStarts.push(now);
  imageJobsInFlight += 1;
}

async function runPictureJob(room, round, words) {
  let imageId = null;
  let failure = null;
  try {
    reserveImageSlot();
    let image;
    try {
      image = await requestImage(words);
    } finally {
      imageJobsInFlight -= 1;
    }
    imageId = storeImage(room, image);
  } catch (error) {
    if (error instanceof ImageError) failure = error.message;
    else {
      console.error("Picture generation failed:", error);
      failure = "The picture service couldn't be reached.";
    }
  }
  if (rooms.get(room.code) !== room) {
    if (imageId) images.delete(imageId);
    return;
  }
  if (imageId && (!room.picture || round > room.picture.round)) {
    room.picture = { round, words, url: `/generated/${imageId}` };
    if (room.pictureJob?.round === round) room.pictureJob = null;
  } else if (imageId) {
    images.delete(imageId);
    room.imageIds = room.imageIds.filter((id) => id !== imageId);
  } else if (room.pictureJob?.round === round) {
    room.pictureJob = { round, words, status: "error", message: failure };
  }
  broadcast(room, "picture:updated");
}

function startPictureJob(room, round, words) {
  room.pictureJob = { round, words, status: "pending", message: null };
  runPictureJob(room, round, words).catch((error) => console.error("Picture job crashed:", error));
}

function completeRound(room) {
  const finishedRound = room.round;
  const finishedWords = room.words;
  room.wordsCompleted += WORDS_PER_ROUND;
  room.score += finishedWords.reduce((total, word) => total + word.length * POINTS_PER_LETTER, 0);
  startPictureJob(room, finishedRound, finishedWords);
  if (room.score >= room.targetScore) {
    finishRoom(room, "score");
    return;
  }
  room.round += 1;
  room.words = pickWords(finishedWords);
  room.letters = ["", ""];
  broadcast(room, "round:completed", { completedWords: finishedWords });
}

function handleLetter(socket, message) {
  const membership = sockets.get(socket);
  if (!membership) return fail(socket, "Join a room before sending a letter.");
  const { room, slot } = membership;
  if (room.status !== "playing") return fail(socket, "The round is not accepting letters right now.");
  if (typeof message.actionId !== "string" || !/^[\da-f-]{16,64}$/i.test(message.actionId)) {
    return fail(socket, "Invalid action identifier.");
  }
  if (room.actionIds.has(message.actionId)) {
    send(socket, { type: "room:updated", room: roomSnapshot(room) });
    return;
  }
  if (typeof message.letter !== "string" || !/^[A-Z]$/.test(message.letter)) {
    return fail(socket, "Send one uppercase letter.");
  }
  const word = room.words[slot];
  const progress = room.letters[slot];
  if (progress.length >= word.length) {
    const partner = room.players[1 - slot]?.name || "your partner";
    return fail(socket, `You've finished ${word}. Waiting for ${partner} to finish their word.`);
  }
  if (message.letter !== word[progress.length]) {
    return fail(socket, "That letter does not match the next letter in your word.");
  }

  room.actionIds.add(message.actionId);
  if (room.actionIds.size > 256) room.actionIds.delete(room.actionIds.values().next().value);
  room.letters[slot] = progress + message.letter;

  if (room.letters.every((letters, index) => letters === room.words[index])) {
    completeRound(room);
    return;
  }
  const completedWord = room.letters[slot] === word ? word : null;
  broadcast(room, "letter:accepted", { accepted: message.letter, slot, completedWord });
}

function handleVideoReady(socket) {
  const membership = sockets.get(socket);
  if (!membership) return fail(socket, "Join a room before enabling video sharing.");
  const { room, slot } = membership;
  const player = room.players[slot];
  if (!player?.connected) return fail(socket, "Your room connection is not ready.");
  player.videoReady = true;
  const peer = room.players[1 - slot];
  if (peer?.connected && peer.videoReady) {
    send(peer.socket, { type: "video:peer-ready", peerSlot: slot });
    send(socket, { type: "video:peer-ready", peerSlot: peer.slot });
  }
  broadcast(room, "room:updated");
  startRoom(room);
}

function handleAslReady(socket, ready) {
  const membership = sockets.get(socket);
  if (!membership) return fail(socket, "Join a room before enabling ASL recognition.");
  const { room, slot } = membership;
  room.players[slot].aslReady = ready;
  broadcast(room, "room:updated");
  if (ready) startRoom(room);
}

function handleVideoSignal(socket, message) {
  const membership = sockets.get(socket);
  if (!membership) return fail(socket, "Join a room before sending video signaling.");
  const { room, slot } = membership;
  const player = room.players[slot];
  const peer = room.players[1 - slot];
  const signal = message.signal;
  if (!player?.videoReady || !peer?.connected || !peer.videoReady) {
    return fail(socket, "Both players must enable video sharing first.");
  }
  if (!signal || typeof signal !== "object" || Array.isArray(signal)) {
    return fail(socket, "Invalid video signaling data.");
  }
  let encoded;
  try {
    encoded = JSON.stringify(signal);
  } catch {
    return fail(socket, "Invalid video signaling data.");
  }
  if (encoded.length > 48_000) return fail(socket, "Video signaling data is too large.");
  if (signal.type === "offer" || signal.type === "answer") {
    if (typeof signal.sdp !== "string" || signal.sdp.length > 44_000) {
      return fail(socket, "Invalid video session description.");
    }
  } else if (signal.type === "candidate") {
    if (signal.candidate !== null && (
      !signal.candidate
      || typeof signal.candidate !== "object"
      || typeof signal.candidate.candidate !== "string"
      || signal.candidate.candidate.length > 4096
    )) {
      return fail(socket, "Invalid network candidate.");
    }
  } else {
    return fail(socket, "Unknown video signaling message.");
  }
  send(peer.socket, { type: "video:signal", fromSlot: slot, signal });
}

function handleMessage(socket, message) {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return fail(socket, "Invalid message.");
  }
  switch (message.type) {
    case "room:create":
      return handleCreate(socket, message);
    case "room:join":
      return handleJoin(socket, message);
    case "letter:confirm":
      return handleLetter(socket, message);
    case "video:ready":
      return handleVideoReady(socket);
    case "asl:ready":
      return handleAslReady(socket, true);
    case "asl:stop":
      return handleAslReady(socket, false);
    case "video:signal":
      return handleVideoSignal(socket, message);
    case "video:stop": {
      const membership = sockets.get(socket);
      if (!membership) return;
      const { room, slot } = membership;
      room.players[slot].videoReady = false;
      const peer = room.players[1 - slot];
      if (peer) {
        peer.videoReady = false;
        if (peer.socket) send(peer.socket, { type: "video:peer-left", peerSlot: slot });
      }
      broadcast(room, "room:updated");
      return;
    }
    case "game:restart":
      return handleRestart(socket, message);
    case "game:leave": {
      detachPlayer(socket);
      socket.close(1000, "Left room");
      return;
    }
    default:
      return fail(socket, "Unknown event.");
  }
}

function serveFile(request, response) {
  let pathname;
  try {
    pathname = new URL(request.url, "http://localhost").pathname;
  } catch {
    response.writeHead(400).end("Bad request");
    return;
  }
  let requested;
  try {
    requested = pathname === "/" ? "/index.html" : decodeURIComponent(pathname);
  } catch {
    response.writeHead(400).end("Bad request");
    return;
  }
  const imageMatch = GENERATED_PATH.exec(requested);
  if (imageMatch) {
    const image = images.get(imageMatch[1]);
    if (!image) {
      response.writeHead(404).end("Not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type": image.type,
      "Content-Length": image.bytes.length,
      "X-Content-Type-Options": "nosniff",
      "Cache-Control": "private, max-age=3600",
    });
    response.end(image.bytes);
    return;
  }
  if (!PUBLIC_FILES.has(requested)) {
    response.writeHead(404).end("Not found");
    return;
  }
  const absolutePath = path.resolve(ROOT, `.${requested}`);
  if (!absolutePath.startsWith(`${ROOT}${path.sep}`)) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  fs.readFile(absolutePath, (error, contents) => {
    if (error) {
      response.writeHead(error.code === "ENOENT" ? 404 : 500).end("Not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type": MIME_TYPES[path.extname(absolutePath)] || "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
      "Cache-Control": path.extname(absolutePath) === ".onnx" ? "public, max-age=3600" : "no-cache",
    });
    response.end(contents);
  });
}

const server = http.createServer(serveFile);
const webSocketServer = new WebSocketServer({ server, maxPayload: 65536 });

webSocketServer.on("connection", (socket, request) => {
  const origin = request.headers.origin;
  if (origin) {
    let sameHost = false;
    try {
      sameHost = new URL(origin).host === request.headers.host;
    } catch {
      sameHost = false;
    }
    if (!sameHost) {
      socket.close(1008, "Cross-origin connections are not allowed.");
      return;
    }
  }
  socket.on("message", (raw) => {
    let message;
    try {
      message = JSON.parse(raw.toString());
    } catch {
      fail(socket, "Message must be valid JSON.");
      return;
    }
    handleMessage(socket, message);
  });

  socket.on("close", () => {
    detachPlayer(socket);
  });
});

setInterval(() => {
  const now = Date.now();
  for (const room of rooms.values()) {
    if (room.status === "playing" && room.deadline <= now) {
      room.remainingMs = 0;
      finishRoom(room, "time");
    } else if (room.status === "playing") {
      broadcast(room, "timer:updated");
    }
    if (room.lastEmptyAt && now - room.lastEmptyAt > 10 * 60 * 1000) {
      forgetImages(room);
      rooms.delete(room.code);
    }
  }
}, 1000).unref();

server.listen(PORT, "0.0.0.0", () => {
  console.log(`SignSync listening on 0.0.0.0:${PORT}`);
});
