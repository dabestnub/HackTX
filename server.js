const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { WebSocketServer, WebSocket } = require("ws");

const ROOT = __dirname;
const PORT = Number(process.env.PORT || 8000);
const ROUND_SECONDS = 120;
const SCORE_PER_WORD = 100;
const WIN_SCORE = 500;
const WORDS = ["CAT", "DOG", "SUN", "HAT", "HOME", "LOVE"];
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

function randomCode() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from(crypto.randomBytes(6), (byte) => alphabet[byte % alphabet.length]).join("");
}

function makeRoom() {
  let code;
  do {
    code = randomCode();
  } while (rooms.has(code));
  const room = {
    code,
    players: [null, null],
    status: "waiting",
    currentPlayer: 0,
    wordIndex: 0,
    letters: "",
    score: 0,
    wordsCompleted: 0,
    deadline: null,
    remainingMs: ROUND_SECONDS * 1000,
    completedReason: null,
    actionIds: new Set(),
    lastEmptyAt: null,
  };
  rooms.set(code, room);
  return room;
}

function currentWord(room) {
  return WORDS[room.wordIndex % WORDS.length];
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
      videoSharing: player.videoReady,
    })),
    currentPlayer: room.currentPlayer,
    target: currentWord(room),
    letters: room.letters,
    score: room.score,
    wordsCompleted: room.wordsCompleted,
    secondsRemaining,
    completedReason: room.completedReason,
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
  if (room.status !== "waiting" || !room.players.every((player) => player?.connected)) return;
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
    videoReady: false,
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

function handleCreate(socket, message) {
  if (sockets.has(socket)) return fail(socket, "Leave your current room before creating another.");
  const name = cleanName(message.name);
  if (!name) return fail(socket, "Enter a nickname up to 24 characters.");
  attachPlayer(socket, makeRoom(), 0, name);
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
  }
  room.lastEmptyAt = room.players.every((entry) => !entry?.connected) ? Date.now() : null;
  broadcast(room, "room:updated");
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
  if (slot !== room.currentPlayer) return fail(socket, "It is the other player's turn.");
  if (typeof message.letter !== "string" || !/^[A-Z]$/.test(message.letter)) {
    return fail(socket, "Send one uppercase letter.");
  }
  const word = currentWord(room);
  if (message.letter !== word[room.letters.length]) {
    return fail(socket, "That letter does not match the next letter in this practice word.");
  }

  room.actionIds.add(message.actionId);
  if (room.actionIds.size > 256) room.actionIds.delete(room.actionIds.values().next().value);
  room.letters += message.letter;
  room.currentPlayer = 1 - room.currentPlayer;

  let completedWord = null;
  if (room.letters === word) {
    completedWord = word;
    room.wordsCompleted += 1;
    room.score = Math.min(WIN_SCORE, room.score + SCORE_PER_WORD);
    room.wordIndex += 1;
    room.letters = "";
  }
  if (room.score >= WIN_SCORE) finishRoom(room, "score");
  else broadcast(room, "letter:accepted", { accepted: message.letter, completedWord });
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
    if (room.lastEmptyAt && now - room.lastEmptyAt > 10 * 60 * 1000) rooms.delete(room.code);
  }
}, 1000).unref();

server.listen(PORT, "0.0.0.0", () => {
  console.log(`SignSync listening on 0.0.0.0:${PORT}`);
});
