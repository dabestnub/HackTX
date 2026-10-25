const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

// Run the inline recognition helpers without camera hardware or CDN downloads.
function loadHelper(name, context) {
  const html = fs.readFileSync('index.html', 'utf8');
  const source = html.match(new RegExp(`      (?:async )?function ${name}\\([^]*?\\n      }`));
  assert.ok(source, `${name} must exist`);
  if (!vm.isContext(context)) vm.createContext(context);
  vm.runInContext(source[0], context);
  return context[name];
}

test('hold takes 1500 ms and resets on changed signs, missing hands, and frame gaps', () => {
  const html = fs.readFileSync('index.html', 'utf8');
  const holdDuration = Number(html.match(/const holdDuration = (\d+);/)[1]);
  assert.equal(holdDuration, 1500);
  for (const id of ['hold-progress', 'lesson-hold-progress']) {
    assert.match(html, new RegExp(`id="${id}" max="1500"`));
  }
  const context = { recognizer: null, holdDuration, holdStartedAt: null, lastHoldAt: null, lastStableLabel: '',
    holdProgress: { value: 0 }, lessonHoldProgress: { value: 0 }, lessonMode: true,
    lessonHoldLabel: { textContent: '' }, holdLabel: { textContent: '' } };
  const hold = loadHelper('updateSignHold', context);
  for (let time = 0; time < 1500; time += 100) assert.equal(hold('C', time), false);
  assert.equal(context.holdProgress.value, 1400);
  assert.equal(context.lessonHoldProgress.value, 1400);
  assert.equal(hold('C', 1499), false);
  assert.equal(hold('C', 1500), true);
  assert.equal(context.holdProgress.value, 1500);
  assert.equal(context.lessonHoldProgress.value, 1500);
  assert.match(context.lessonHoldLabel.textContent, /1\.5 \/ 1\.5 seconds/);
  assert.match(context.holdLabel.textContent, /1\.5 \/ 1\.5 seconds/);
  assert.equal(hold('A', 1600), false);
  assert.equal(context.holdProgress.value, 0);
  hold('', 1700);
  assert.equal(hold('A', 1800), false);
  assert.equal(hold('A', 4000), false);
  assert.equal(context.holdProgress.value, 0);
});

test('random lesson order contains each alphabet letter once', () => {
  const context = { lessonAlphabet: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', Math: { random: () => 0, floor: Math.floor } };
  const shuffled = loadHelper('shuffledLessonAlphabet', context)();
  assert.equal(shuffled.length, 26);
  assert.deepEqual([...shuffled].sort(), [...context.lessonAlphabet].sort());
  assert.notEqual(shuffled.join(''), context.lessonAlphabet);
});

test('automatic submission validates the local word and prevents duplicate sends', () => {
  const sent = [];
  const context = { roomState: { status: 'playing', words: ['CAT', 'DOG'], letters: ['', ''] }, playerSlot: 0,
    debugPreview: false, submittingLetter: false, socket: {readyState: 1, send: data => sent.push(JSON.parse(data))},
    WebSocket: {OPEN: 1}, select: {value: 'B'}, target: 'CAT', accepted: '', notice: {},
    predictionSource: 'camera', pendingSource: '', currentPrediction: 'C', waitingForRelease: false,
    livePrediction: {}, predictionDetail: {}, crypto: {getRandomValues: array => require('node:crypto').randomFillSync(array)}, Uint8Array, render() {} };
  context.canSign = loadHelper('canSign', context);
  const submit = loadHelper('submitLetter', context);
  submit();
  assert.equal(sent.length, 0);
  context.select.value = 'C';
  submit();
  assert.equal(sent.length, 1);
  assert.equal(sent[0].letter, 'C');
  assert.match(sent[0].actionId, /^[a-f0-9]{32}$/);
  assert.equal(context.waitingForRelease, true);
  context.select.value = 'C';
  submit();
  assert.equal(sent.length, 1);
  context.submittingLetter = false;
  context.select.value = 'C';
  context.socket.send = () => { throw new Error('Disconnected'); };
  submit();
  assert.equal(context.submittingLetter, false);
  assert.equal(context.select.value, 'C');
  assert.match(context.notice.textContent, /try again/i);
  context.roomState.letters[0] = 'CAT';
  submit();
  assert.equal(sent.length, 1);
});


test('manual letters submit with the camera enabled while the keypad is open', () => {
  let submissions = 0;
  const context = { debugPreview: false, submittingLetter: false, cameraStream: {},
    roomState: {status: 'playing', words: ['CAT', 'DOG'], letters: ['', '']}, playerSlot: 0,
    manualInput: {open: true}, select: {value: ''}, currentPrediction: '', predictionSource: '',
    livePrediction: {}, predictionDetail: {}, notice: {},
    submitLetter() { submissions += 1; } };
  context.canSign = loadHelper('canSign', context);
  const enter = loadHelper('enterManualLetter', context);
  enter('C');
  assert.equal(submissions, 1);
  assert.equal(context.select.value, 'C');
  assert.equal(context.predictionSource, 'manual');
  context.roomState.letters[0] = 'CAT';
  enter('A');
  assert.equal(submissions, 1);
  context.manualInput.open = false;
  enter('A');
  assert.equal(submissions, 1);
});

test('copy uses the room code and offers selectable text when clipboard access fails', async () => {
  let copied;
  let selected = false;
  const context = {roomState: {code: 'ABC234'},
    navigator: {clipboard: {writeText: async text => { copied = text; }}},
    copyButton: {}, copyStatus: {}, copyFallback: {hidden: true, focus() {}, select() { selected = true; }},
    document: {execCommand: () => false}};
  const copy = loadHelper('copyRoomCode', context);
  await copy();
  assert.equal(copied, 'ABC234');
  assert.match(context.copyStatus.textContent, /copied/i);
  context.navigator.clipboard = undefined;
  await copy();
  assert.equal(context.copyFallback.value, 'ABC234');
  assert.equal(context.copyFallback.hidden, false);
  assert.equal(selected, true);
  assert.match(context.copyStatus.textContent, /copy.*manually/i);
});

test('each player sees their own word and cannot sign after finishing it', () => {
  const node = () => ({ style: {}, classList: {toggle() {}}, append() {}, replaceChildren() {}, setAttribute() {},
    getAttribute() { return null; }, removeAttribute() {}, focus() {} });
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); };
  const context = {
    document: {querySelector: get, createElement: node, body: { classList: { remove() {} } }},
    lessonMode: true,
    roomState: {status: 'playing', code: 'ABC234', words: ['CAT', 'DOG'], letters: ['', ''], score: 0,
      targetScore: 500, roundDuration: 120, wordsCompleted: 0, secondsRemaining: 120, pendingRematch: null,
      picture: null, pictureJob: null, players: [{name: 'Alex', connected: true}, {name: 'Sam', connected: true}]},
    playerSlot: 0, debugPreview: false, submittingLetter: false, cameraStream: {},
    socket: {readyState: 1}, WebSocket: {OPEN: 1}, peerConnection: null, videoShareStream: null,
    target: 'CAT', accepted: '', debugKeypad: {children: [node()]},
    updateSignHold() {}, renderResults() {}, updateGameTheme() {},
  };
  context.canSign = loadHelper('canSign', context);
  context.buildSlots = loadHelper('buildSlots', context);
  for (const key of ['menuScreen', 'waitingScreen', 'gameScreen', 'localMediaPanel', 'remoteVideoPreview',
    'manualStatus', 'select', 'resetButton', 'cameraStart', 'videoShareStart', 'videoShareStop', 'videoShareStatus']) context[key] = node();
  const render = loadHelper('updateRoomUI', context);
  render();
  assert.equal(context.lessonMode, false);
  assert.equal(get('#turn-warning').hidden, false);
  assert.match(get('#turn-warning-copy').textContent, /Sign C/);
  assert.equal(get('#partner-label').textContent, "Sam's word: DOG");
  assert.equal(context.debugKeypad.children[0].disabled, false);
  assert.match(context.manualStatus.textContent, /click C/i);
  context.roomState.letters[0] = 'CAT';
  render();
  assert.equal(get('#turn-warning').hidden, false);
  assert.match(get('#turn-warning-copy').textContent, /finished CAT.*Sam/);
  assert.equal(context.debugKeypad.children[0].disabled, true);
  context.roomState.showPartner = false;
  context.roomState.words[1] = '';
  context.roomState.picture = {words: ['CAT', ''], url: '/generated/test'};
  render();
  assert.equal(get('#player-two').hidden, true);
  assert.equal(get('#partner-label').hidden, true);
  assert.equal(get('#partner-letters').hidden, true);
  assert.equal(context.videoShareStart.hidden, true);
  assert.match(get('#image-caption').textContent, /mystery/);
});

test('results keep player cameras visible after timeout and show the final score', () => {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, {focus() {}}); return elements.get(id); };
  const context = {renderGallery() {}, document: {querySelector: get}, resultsScreen: {hidden: true}, gameScreen: {hidden: false, classList: {toggle() {}}},
    socket: {readyState: 1}, WebSocket: {OPEN: 1, CONNECTING: 0},
    playerSlot: 0,
    roomState: {status: 'complete', completedReason: 'time', score: 300, wordsCompleted: 3, targetScore: 750,
      roundDuration: 120, pendingRematch: null, players: [{name: 'Alex', rematchReady: false}, {name: 'Sam', rematchReady: false}]}};
  const show = loadHelper('renderResults', context);
  show();
  assert.equal(context.resultsScreen.hidden, false);
  assert.equal(context.gameScreen.hidden, false);
  assert.equal(get('#results-title').textContent, 'Time’s up!');
  assert.equal(get('#results-score').textContent, 300);
  assert.equal(get('#results-words').textContent, 3);
  context.roomState.completedReason = 'score';
  show();
  assert.equal(get('#results-title').textContent, '750 points. Together.');
  context.roomState.status = 'playing';
  show();
  assert.equal(context.resultsScreen.hidden, true);
});

test('reconnect reuses the room and player token and guards repeated clicks', () => {
  const sent = [];
  const context = {roomState: {code: 'ABC234', players: [{name: 'Alex'}]}, playerSlot: 0,
    playerToken: 'saved-token', debugPreview: false, socket: null, WebSocket: {CONNECTING: 0, OPEN: 1},
    reconnectButton: {}, reconnectStatus: {}, leavingRoom: true,
    connectAndSend(payload) { sent.push(payload); }, showLobbyError() {}};
  const reconnect = loadHelper('reconnectRoom', context);
  reconnect();
  assert.equal(sent[0].type, 'room:join');
  assert.equal(sent[0].code, 'ABC234');
  assert.equal(sent[0].token, 'saved-token');
  assert.equal(sent[0].name, 'Alex');
  context.socket = {readyState: 0};
  reconnect();
  assert.equal(sent.length, 1);
});


test('game completion keeps the local camera and shared video running', () => {
  let stops = 0;
  const context = {submittingLetter: true, roomState: null, target: '', accepted: '', playerSlot: 0,
    cameraStream: {}, videoShareStream: {}, peerConnection: {},
    stopCamera() { stops += 1; }, stopVideoShare() { stops += 1; },
    showLobbyError() {}, updateRoomUI() {}, document: {querySelector: () => ({})}};
  const handle = loadHelper('handleSocketMessage', context);
  handle({data: JSON.stringify({type: 'game:completed', room: {
    status: 'complete', completedReason: 'time', words: ['CAT', 'DOG'], letters: ['', ''], targetScore: 500, roundDuration: 120,
  }})});
  assert.equal(stops, 0);
  assert.equal(context.roomState.status, 'complete');
});

test('results gallery keeps drafts while pictures arrive and only hidden games offer guesses', () => {
  const node = tag => ({ tag, children: [], dataset: {}, listeners: {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute() {}, addEventListener(type, listener) { this.listeners[type] = listener; },
    querySelector(selector) {
      if (selector === 'input') return this.children.find(child => child.tag === 'input') || this.children.map(child => child.querySelector(selector)).find(Boolean);
      return this.children.find(child => String(child.dataset.round) === selector.match(/"(\d+)"/)[1]);
    } });
  const gallery = node('div'), instructions = node('p');
  const sent = [];
  const context = { document: {createElement: node, querySelector: id => id === '#results-gallery' ? gallery : instructions},
    roomState: {code: 'ABC234', gameId: 1, showPartner: false, pictures: [
      {round: 1, status: 'ready', url: '/one', words: ['CAT', ''], guess: null},
      {round: 2, status: 'pending', words: ['SUN', ''], guess: null}]},
    playerSlot: 0, socket: {readyState: 1, send: data => sent.push(JSON.parse(data))}, WebSocket: {OPEN: 1} };
  const render = loadHelper('renderGallery', context);
  const descendants = parent => parent.children.flatMap(child => [child, ...descendants(child)]);
  render();
  assert.equal(gallery.children.length, 2);
  const input = descendants(gallery).find(child => child.tag === 'input');
  input.value = 'DOG';
  context.roomState.pictures[1].url = '/two';
  context.roomState.pictures[1].status = 'ready';
  render();
  assert.equal(descendants(gallery).find(child => child.tag === 'input'), input);
  assert.equal(descendants(gallery).filter(child => child.tag === 'img').length, 2);
  descendants(gallery).find(child => child.tag === 'form').listeners.submit({preventDefault() {}});
  assert.deepEqual(sent, [{type: 'picture:guess', round: 1, guess: 'DOG'}]);
  Object.assign(context.roomState.pictures[0], {guess: 'DOG', correct: true, words: ['CAT', 'DOG']});
  render();
  assert.ok(descendants(gallery.children[0]).some(child => /Correct!/.test(child.textContent)));
  context.roomState.gameId = 2;
  context.roomState.showPartner = true;
  render();
  assert.equal(descendants(gallery).filter(child => child.tag === 'form').length, 0);
  context.roomState.gameId = 3;
  context.roomState.pictures = [];
  render();
  assert.equal(gallery.children.length, 0);
  assert.match(instructions.textContent, /No pictures/);
});

test('stopping video in hidden mode does not offer sharing again', async () => {
  let stopped = false;
  const context = {roomState: {showPartner: false}, videoShareStream: {getTracks: () => [{stop() { stopped = true; }}]},
    socket: null, WebSocket: {OPEN: 1}, closePeerConnection() {}, setVideoShareStatus() {},
    videoShareStart: {}, videoShareStop: {}};
  loadHelper('stopVideoShare', context)(false);
  assert.equal(stopped, true);
  assert.equal(context.videoShareStart.hidden, true);
  assert.equal(context.videoShareStart.disabled, true);
  // A direct invocation must not even request camera permission in hidden mode.
  await loadHelper('startVideoShare', context)();
  assert.equal(context.videoShareStream, null);
  context.roomState.showPartner = true;
  context.videoShareStream = null;
  context.stopVideoShare(false);
  assert.equal(context.videoShareStart.hidden, false);
  assert.equal(context.videoShareStart.disabled, false);
});

test('opening manual input marks the player ready without enabling their camera', () => {
  const sent = [];
  const context = {debugPreview: false, roomState: {showPartner: false}, socket: {readyState: 1, send: data => sent.push(JSON.parse(data))},
    WebSocket: {OPEN: 1}, manualInput: {open: true}, cameraStream: null, updateSignHold() {}, render() {}};
  const toggle = loadHelper('updateManualInput', context);
  toggle();
  context.manualInput.open = false;
  toggle();
  assert.deepEqual(sent, [{type: 'manual:ready', ready: true}, {type: 'manual:ready', ready: false}]);
});

test('completed word pairs change the palette once, reset for new games, and respect reduced motion', () => {
  let animations = 0, reduced = false;
  const label = {textContent: '', animate() { animations++; }};
  const context = {document: {body: {dataset: {}}, querySelector: () => label},
    window: {matchMedia: () => ({matches: reduced})}};
  const update = loadHelper('updateGameTheme', context);
  update(0, 'room:1');
  assert.equal(context.document.body.dataset.palette, 'blue');
  update(2, 'room:1');
  assert.equal(context.document.body.dataset.palette, 'violet');
  assert.match(label.textContent, /2 words/);
  assert.equal(animations, 1);
  update(2, 'room:1');
  assert.equal(animations, 1);
  reduced = true;
  update(4, 'room:1');
  assert.equal(context.document.body.dataset.palette, 'mint');
  assert.equal(animations, 1);
  update(6, 'room:1');
  assert.equal(context.document.body.dataset.palette, 'orange');
  update(8, 'room:1');
  assert.equal(context.document.body.dataset.palette, 'blue');
  update(0, 'room:2');
  assert.equal(context.document.body.dataset.palette, 'blue');
  assert.equal(animations, 1);
});

test('practice camera controls are mounted without starting camera capture', () => {
  const panel = {};
  const context = { localMediaPanel: panel, cameraStart: {},
    lessonCameraSlot: { append(value) { this.child = value; } } };
  loadHelper('showPracticeCamera', context)();
  assert.equal(context.lessonCameraSlot.child, panel);
  assert.equal(context.cameraStart.textContent, 'Start lesson camera');
});
