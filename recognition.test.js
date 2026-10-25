const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

// Run the inline recognition helpers without camera hardware or CDN downloads.
function loadHelper(name, context) {
  const html = fs.readFileSync('index.html', 'utf8');
  const source = html.match(new RegExp(`      (?:async )?function ${name}\\([^]*?\\n      }`));
  assert.ok(source, `${name} must exist`);
  vm.createContext(context);
  vm.runInContext(source[0], context);
  return context[name];
}

test('hold takes 1500 ms and resets on changed signs, missing hands, and frame gaps', () => {
  const context = { holdDuration: 1500, holdStartedAt: null, lastHoldAt: null, lastStableLabel: '',
    holdProgress: { value: 0 }, holdLabel: { textContent: '' } };
  const hold = loadHelper('updateSignHold', context);
  for (let time = 0; time < 1500; time += 100) assert.equal(hold('C', time), false);
  assert.equal(context.holdProgress.value, 1400);
  assert.equal(hold('C', 1500), true);
  assert.equal(context.holdProgress.value, 1500);
  assert.equal(hold('A', 1600), false);
  assert.equal(context.holdProgress.value, 0);
  hold('', 1700);
  assert.equal(hold('A', 1800), false);
  assert.equal(hold('A', 4000), false);
  assert.equal(context.holdProgress.value, 0);
});

test('automatic submission validates turns and letters and prevents duplicate sends', () => {
  const sent = [];
  const context = { roomState: { status: 'playing', words: ['CAT', 'DOG'], letters: ['', ''] }, playerSlot: 0,
    debugPreview: false, submittingLetter: false, socket: {readyState: 1, send: data => sent.push(JSON.parse(data))},
    WebSocket: {OPEN: 1}, select: {value: 'B'}, target: 'CAT', accepted: '', notice: {},
    predictionSource: 'camera', pendingSource: '', currentPrediction: 'C', waitingForRelease: false,
    livePrediction: {}, predictionDetail: {},
    crypto: {getRandomValues: array => require('node:crypto').randomFillSync(array)}, Uint8Array, render() {} };
  context.canSign = () => context.roomState.status === 'playing'
    && context.roomState.letters[context.playerSlot].length < context.roomState.words[context.playerSlot].length;
  const submit = loadHelper('submitLetter', context);
  submit();
  assert.equal(sent.length, 0);
  context.select.value = 'C';
  context.roomState.letters[0] = 'CAT';
  submit();
  assert.equal(sent.length, 0);
  context.roomState.letters[0] = '';
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
});


test('manual letters submit with the camera enabled while the keypad is open', () => {
  let submissions = 0;
  const context = { debugPreview: false, submittingLetter: false, cameraStream: {},
    roomState: {status: 'playing', words: ['CAT', 'DOG'], letters: ['', '']}, playerSlot: 0,
    manualInput: {open: true}, select: {value: ''}, currentPrediction: '', predictionSource: '',
    livePrediction: {}, predictionDetail: {}, notice: {},
    submitLetter() { submissions += 1; } };
  context.canSign = () => context.roomState.status === 'playing'
    && context.roomState.letters[context.playerSlot].length < context.roomState.words[context.playerSlot].length;
  const enter = loadHelper('enterManualLetter', context);
  enter('C');
  assert.equal(submissions, 1);
  assert.equal(context.select.value, 'C');
  assert.equal(context.predictionSource, 'manual');
  context.roomState.letters[0] = 'CAT';
  enter('A');
  assert.equal(submissions, 1);
  context.roomState.letters[0] = '';
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

test('turn warning and manual keypad follow the local player even with a camera on', () => {
  const node = () => ({
    style: {},
    classList: {toggle() {}, remove() {}},
    append() {},
    replaceChildren() {},
    setAttribute() {},
    removeAttribute() {},
    getAttribute() { return ''; },
  });
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); };
  const context = {
    document: {querySelector: get, createElement: node},
    roomState: {status: 'playing', code: 'ABC234', words: ['CAT', 'DOG'], letters: ['', ''], score: 0,
      secondsRemaining: 120, players: [{name: 'Alex', connected: true}, {name: 'Sam', connected: true}], picture: null, pictureJob: null},
    playerSlot: 0, debugPreview: false, submittingLetter: false, cameraStream: {},
    socket: {readyState: 1}, WebSocket: {OPEN: 1}, peerConnection: null, videoShareStream: null,
    target: 'CAT', accepted: '', debugKeypad: {children: [node()]},
    updateSignHold() {}, renderResults() {},
  };
  context.canSign = () => context.roomState.status === 'playing'
    && context.roomState.letters[context.playerSlot].length < context.roomState.words[context.playerSlot].length;
  for (const key of ['menuScreen', 'waitingScreen', 'gameScreen', 'localMediaPanel', 'remoteVideoPreview',
    'manualStatus', 'select', 'resetButton', 'cameraStart', 'videoShareStart']) context[key] = node();
  const render = loadHelper('updateRoomUI', context);
  render();
  assert.equal(get('#turn-warning').hidden, false);
  assert.equal(get('#turn-warning-title').textContent, 'SIGN YOUR WORD');
  assert.equal(context.debugKeypad.children[0].disabled, false);
  context.roomState.letters[0] = 'CAT';
  render();
  assert.equal(get('#turn-warning').hidden, false);
  assert.equal(get('#turn-warning-title').textContent, 'WORD DONE');
  assert.equal(context.debugKeypad.children[0].disabled, true);
  assert.match(context.manualStatus.textContent, /waiting for/i);
  context.roomState.pictureJob = {status: 'pending', words: ['CAT', 'DOG']};
  context.roomState.score = 100;
  render({type: 'round:completed', completedWords: ['CAT', 'DOG']});
  assert.equal(get('#turn-warning').hidden, false);
  assert.equal(get('#image-copy').textContent.includes('Making a picture'), true);
  context.roomState.status = 'complete';
  render({type: 'game:completed'});
  assert.equal(get('#turn-warning').hidden, false);
  assert.equal(get('#turn-warning-title').textContent, 'ROUND COMPLETE');
});

test('results keep player cameras visible after timeout and show the final score', () => {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, {focus() {}}); return elements.get(id); };
  const context = {document: {querySelector: get}, resultsScreen: {hidden: true}, gameScreen: {hidden: false, classList: {toggle() {}}},
    roomState: {status: 'complete', completedReason: 'time', score: 300, wordsCompleted: 3}};
  const show = loadHelper('renderResults', context);
  show();
  assert.equal(context.resultsScreen.hidden, false);
  assert.equal(context.gameScreen.hidden, false);
  assert.equal(get('#results-title').textContent, 'Time’s up!');
  assert.equal(get('#results-score').textContent, 300);
  assert.equal(get('#results-words').textContent, 3);
  context.roomState.completedReason = 'score';
  show();
  assert.equal(get('#results-title').textContent, '500 points. Together.');
  context.roomState.status = 'playing';
  show();
  assert.equal(context.resultsScreen.hidden, true);
});

test('reconnect reuses the room and player token and guards repeated clicks', () => {
  const sent = [];
  const context = {roomState: {code: 'ABC234', players: [{name: 'Alex'}]}, playerSlot: 0,
    playerToken: 'saved-token', debugPreview: false, socket: null, WebSocket: {CONNECTING: 0},
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


test('round completion keeps the local camera and shared video running', () => {
  let stops = 0;
  const context = {submittingLetter: true, roomState: null, target: '', accepted: '', playerSlot: 0,
    cameraStream: {}, videoShareStream: {}, peerConnection: {},
    stopCamera() { stops += 1; }, stopVideoShare() { stops += 1; },
    showLobbyError() {}, updateRoomUI() {}, document: {querySelector: () => ({})}};
  const handle = loadHelper('handleSocketMessage', context);
  handle({data: JSON.stringify({type: 'game:completed', room: {
    status: 'complete', completedReason: 'time', words: ['CAT', 'DOG'], letters: ['CAT', 'DOG'],
  }})});
  assert.equal(stops, 0);
  assert.equal(context.roomState.status, 'complete');
});
