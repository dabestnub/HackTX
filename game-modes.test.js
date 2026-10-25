const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

function loadGame() {
  const context = vm.createContext({ require, __dirname, process, console, Buffer, URL, setTimeout, clearTimeout });
  vm.runInContext(fs.readFileSync('server.js', 'utf8').split('const server = http.createServer')[0], context);
  return context;
}

function socket() {
  return { readyState: 1, messages: [], send(data) { this.messages.push(JSON.parse(data)); } };
}

test('hidden rooms redact partner words and events and start without shared video', () => {
  const game = loadGame();
  const room = game.makeRoom(500, 120, false);
  const a = socket(), b = socket();
  game.attachPlayer(a, room, 0, 'Alex');
  game.attachPlayer(b, room, 1, 'Sam');
  assert.equal(a.messages[0].room.words[1], '');
  assert.equal(b.messages[0].room.words[0], '');
  game.handleAslReady(a, true);
  game.handleAslReady(b, true);
  assert.equal(room.status, 'playing');
  game.handleLetter(a, { actionId: '0000000000000001', letter: room.words[0][0] });
  assert.equal(b.messages.at(-1).accepted, undefined);
  assert.equal(b.messages.at(-1).room.letters[0], '');
  game.broadcast(room, 'round:completed', { completedWords: ['CAT', 'DOG'] });
  assert.equal(b.messages.at(-1).completedWords, undefined);
  game.handleVideoReady(a);
  assert.equal(room.players[0].videoReady, false);
  assert.match(a.messages.at(-1).message, /hidden/i);
});

test('gallery preserves every image including out-of-order jobs and reveals only after guessing', async () => {
  const game = loadGame();
  const room = game.makeRoom(500, 120, false);
  const a = socket(), b = socket();
  game.attachPlayer(a, room, 0, 'Alex');
  game.attachPlayer(b, room, 1, 'Sam');
  const pending = [];
  game.requestImage = () => new Promise(resolve => pending.push(resolve));
  const image = { type: 'image/png', bytes: Buffer.from('picture') };
  game.startPictureJob(room, 1, ['CAT', 'DOG']);
  game.startPictureJob(room, 2, ['SUN', 'HAT']);
  pending[1](image);
  await new Promise(setImmediate);
  pending[0](image);
  await new Promise(setImmediate);
  for (let round = 3; round <= 6; round++) {
    game.startPictureJob(room, round, ['CAT', 'DOG']);
    pending.at(-1)(image);
    await new Promise(setImmediate);
  }
  assert.equal(room.pictures.length, 6);
  assert.equal(room.imageIds.length, 6);
  assert.equal(room.picture.round, 6);
  assert.equal(game.roomSnapshot(room, 0).pictures[0].words[1], '');
  game.handleGuess(a, { round: 1, guess: 'dog' });
  assert.equal(a.messages.at(-1).type, 'error');
  game.finishRoom(room, 'time');
  game.handleGuess(a, { round: 1, guess: ' dog ' });
  const revealed = game.roomSnapshot(room, 0).pictures[0];
  assert.equal(revealed.words[1], 'DOG');
  assert.equal(revealed.guess, 'DOG');
  assert.equal(revealed.correct, true);
  assert.equal(game.roomSnapshot(room, 1).pictures[0].words[0], '');
  game.startPictureJob(room, 7, ['SUN', 'HAT']);
  game.handleRestart(a, {});
  game.handleRestart(b, {});
  pending.at(-1)(image);
  await new Promise(setImmediate);
  assert.equal(room.pictures.length, 0);
  assert.equal(room.imageIds.length, 0);
  assert.equal(room.picture, null);
});

test('visible rooms retain both words and do not accept guesses', () => {
  const game = loadGame();
  const room = game.makeRoom(500);
  const a = socket();
  game.attachPlayer(a, room, 0, 'Alex');
  assert.deepEqual(Array.from(game.roomSnapshot(room, 0).words), Array.from(room.words));
  assert.equal(room.showPartner, true);
  game.finishRoom(room, 'time');
  game.handleGuess(a, { round: 1, guess: 'CAT' });
  assert.equal(a.messages.at(-1).type, 'error');
});

test('wrong guesses generate a private hint once and allow a correct retry', async () => {
  const game = loadGame();
  const room = game.makeRoom(500, 120, false);
  const a = socket(), b = socket();
  game.attachPlayer(a, room, 0, 'Alex');
  game.attachPlayer(b, room, 1, 'Sam');
  room.pictures.push({round: 1, words: ['CAT', 'DOG'], url: '/picture', guesses: [null, null], hints: [null, null]});
  game.finishRoom(room, 'time');
  let requests = 0, resolveHint;
  game.requestHint = async word => { assert.equal(word, 'DOG'); requests++; return new Promise(resolve => {resolveHint = resolve;}); };
  game.handleGuess(a, {round: 1, guess: 'FISH'});
  assert.equal(game.roomSnapshot(room, 0).pictures[0].words[1], '');
  assert.equal(game.roomSnapshot(room, 0).pictures[0].hint.status, 'pending');
  game.handleGuess(a, {round: 1, guess: 'BIRD'});
  assert.equal(requests, 1);
  resolveHint('This pet barks and wags its tail.');
  await new Promise(setImmediate);
  assert.equal(game.roomSnapshot(room, 0).pictures[0].hint.text, 'This pet barks and wags its tail.');
  assert.equal(game.roomSnapshot(room, 1).pictures[0].hint, null);
  game.handleGuess(a, {round: 1, guess: 'DOG'});
  assert.equal(game.roomSnapshot(room, 0).pictures[0].correct, true);
  assert.equal(game.roomSnapshot(room, 0).pictures[0].words[1], 'DOG');
  assert.equal(requests, 1);
});

test('hint API accepts usable clues and rejects errors, empty text, and answer leaks', async () => {
  const game = loadGame();
  game.process = {env: {META_MUSE_API_KEY: 'test-key'}};
  game.AbortSignal = AbortSignal;
  game.fetch = async (url, options) => {
    assert.equal(url, 'https://api.meta.ai/v1/chat/completions');
    assert.equal(options.headers.Authorization, 'Bearer test-key');
    assert.match(JSON.parse(options.body).messages[1].content, /DOG/);
    return {ok: true, json: async () => ({choices: [{message: {content: 'A pet that barks.'}}]})};
  };
  assert.equal(await game.requestHint('DOG'), 'A pet that barks.');
  for (const content of ['', 'The answer is dog.', null]) {
    game.fetch = async () => ({ok: true, json: async () => ({choices: [{message: {content}}]})});
    await assert.rejects(game.requestHint('DOG'), /unusable/);
  }
  game.fetch = async () => ({ok: false, status: 429});
  await assert.rejects(game.requestHint('DOG'), /429/);
});

test('hint failures keep the answer private and allow further guesses', async () => {
  const game = loadGame();
  const room = game.makeRoom(500, 120, false);
  const a = socket();
  game.attachPlayer(a, room, 0, 'Alex');
  room.pictures.push({round: 1, words: ['CAT', 'DOG'], url: '/picture', guesses: [null, null], hints: [null, null]});
  game.finishRoom(room, 'time');
  game.requestHint = async () => { throw new Error('provider failure'); };
  game.handleGuess(a, {round: 1, guess: 'FISH'});
  await new Promise(setImmediate);
  assert.equal(game.roomSnapshot(room, 0).pictures[0].hint.status, 'error');
  assert.equal(game.roomSnapshot(room, 0).pictures[0].words[1], '');
  game.handleGuess(a, {round: 1, guess: 'DOG'});
  assert.equal(game.roomSnapshot(room, 0).pictures[0].correct, true);
});
