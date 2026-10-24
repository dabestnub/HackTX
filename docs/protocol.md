# ASL Co-op WebSocket Protocol

This is the frontend contract for the current backend (`server/app.py`, `server/game.py`, and production defaults in `server/config.py`). JSON messages are WebSocket text messages; camera JPEGs are binary messages. There is no production typed-letter message or recognition bypass.

## Endpoints

| What | Production | Local |
|---|---|---|
| WebSocket | `wss://<app>.fly.dev/ws` | `ws://localhost:8000/ws` |
| Images | `https://<app>.fly.dev` + the `url` in `image_ready` | `http://localhost:8000` + `url` |
| Health | `GET /healthz` returns `{"ok": true, "rooms": N}` | same |
| Test page | `GET /dev` | same |

The frontend runs on Vercel and connects directly to the Fly backend. Image URLs are relative paths such as `/images/ABCD/1`. Resolve them against the HTTP(S) backend origin, never the Vercel origin: `new URL(msg.url, BACKEND_ORIGIN)`. Use the resulting URL for `<img>` or `fetch()`; the backend allows HTTP requests from all origins. Treat image URLs and IDs as opaque: IDs are unique for the room lifetime, including across `play_again`, and are not round numbers.

Run exactly one Fly machine and one backend process/worker: all rooms, tokens, images, and state live in that process memory. Restarting the backend loses them. Missing or expired image URLs return HTTP 404. `/healthz` reports process health and room count, not model or Muse readiness.

## Game flow

1. Open the socket. The first message must be `create_room`, `join_room`, or `rejoin`.
2. The server answers with `room_joined`. Store `room` and `token`.
3. Both players send `ready`. A 3-second `countdown` starts.
4. `round_start` gives each player a word. The phase becomes `signing`, and the team has 10 seconds in total to finish both words.
5. While signing, send camera frames as binary messages. You get a `recognition` for each processed frame, and both players get a `progress` each time a letter is accepted.
6. When both words are done, everyone gets `image_pending`, then `image_ready` and `round_complete` (+100 points). Below 500 points, the image shows for 5 seconds, then the next countdown starts. The final image goes directly to `game_over`.
7. On timeout, everyone gets `round_failed` (0 points) and a new countdown with a fresh pair starts.
8. At 500 points, everyone gets `game_over` with the gallery. Either player can send `play_again` to return to the lobby with the score reset.

Phases, as sent in `room_state.phase`: `lobby`, `countdown`, `signing`, `generating`, `reveal`, `game_over`.

## Camera frames (client to server, binary)

- One JPEG image per binary WebSocket message.
- Send the **raw, unmirrored** camera image. Mirror only your on-screen preview (CSS `transform: scaleX(-1)`).
- At most 640 pixels wide and 480 high, at most **300,000 bytes** (exactly 300,000 is allowed), at most 10 frames per second. JPEG quality 0.7 is plenty. These are decimal bytes, not 300 KiB.
- Send frames only while `phase` is `signing`, `paused` is false, and your word is unfinished. The server ignores frames outside these conditions or after the round deadline.
- The server retains only the newest waiting frame per player and processes at most 10 per second. Frames can be dropped, so do not expect one reply per sent frame. Also avoid building a browser-side send backlog (`ws.bufferedAmount`).

```js
canvas.width = 640; canvas.height = 480;
canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
canvas.toBlob((blob) => {
  if (blob && blob.size <= 300000 && ws.readyState === WebSocket.OPEN &&
      phase === "signing" && !paused && !wordDone && ws.bufferedAmount === 0) {
    ws.send(blob);
  }
}, "image/jpeg", 0.7);
```

Capture at intervals of at least 100 ms. Do not flip or transform the canvas context; mirror the preview only.

## Client to server (JSON text)

| Message | When |
|---|---|
| `{"type":"create_room","name":"Ann"}` | First message, to start a new room. Names are truncated to 20 characters; omitted/empty names become `Player`. |
| `{"type":"join_room","room":"ABCD","name":"Ben"}` | First message, to join a friend's room. The code is trimmed and not case-sensitive. |
| `{"type":"rejoin","room":"ABCD","token":"..."}` | First message after a dropped connection. Use before the room closes, 30 seconds after the first outstanding disconnect. |
| `{"type":"ready","ready":true}` | In the lobby. Send `false` to un-ready; omitted `ready` defaults to true. Use a JSON boolean. Ignored in other phases. |
| `{"type":"play_again"}` | Either player may send this after `game_over`. Ignored in other phases. |

## Server to client (JSON text)

`room_joined`, sent only to you:
```json
{"type":"room_joined","room":"ABCD","slot":1,"token":"x7..."}
```
Keep `token` for `rejoin`. `slot` is 1 or 2.

`room_state`, broadcast after membership and phase changes and lobby ready updates (not every frame, accepted letter, or timer tick):
```json
{"type":"room_state","room":"ABCD","phase":"signing","paused":false,"score":100,"target":500,"round":2,
 "time_left":10.0,
 "players":[{"slot":1,"name":"Ann","ready":true,"connected":true},
            {"slot":2,"name":"Ben","ready":true,"connected":true}]}
```
`time_left` is the number of seconds left in the round during `signing`, and `null` otherwise. Synchronize the round timer display from it and count down locally only while unpaused; this is a snapshot, not a periodic timer update. `paused` is true while a player is disconnected. Phase timers freeze until all players return; image generation can still finish while paused. `players` contains only occupied slots. During `countdown`, `room_state.round` is the last started round, while `countdown.round` identifies the upcoming round.

`countdown`:
```json
{"type":"countdown","round":2,"seconds":3.0}
```

`round_start`, sent to each player with their own `your_word`:
```json
{"type":"round_start","round":2,"phrase":"BIG DOG","words":[{"slot":1,"word":"DOG"},{"slot":2,"word":"BIG"}],
 "your_word":"DOG","time_limit":10.0}
```
Words are 3 to 5 uppercase letters, never J or Z. Slot 1 signs the adjective on odd rounds and the noun on even rounds; slot 2 signs the other word. The phrase always stays adjective then noun. Server validation avoids words already used in the current game; after exhausting the built-in fallback pool, fallback words may repeat.

`recognition`, sent only to the signer, once per processed frame (up to 10 per second):
```json
{"type":"recognition","guess":"D","confidence":0.93,"hold":0.5,"accepted":false,"expected":"D",
 "landmarks":null}
```
- `guess` is the model's top letter, or `null` when no hand is seen. `confidence` runs from 0 to 1.
- `expected` is the letter evaluated for this frame, including when `accepted` is true; use `progress.done` to advance the word UI. J and Z are masked from guesses, and confidence is the raw softmax value without renormalizing.
- A letter is accepted when `guess` equals `expected` with confidence of at least 0.8, held for 0.6 seconds. `hold` is that hold's progress, from 0 to 1. Wrong letters carry no penalty. No hand, a wrong guess, or confidence below 0.8 resets the hold.
- For a double letter such as BOOK, keep holding. The second O needs its own full hold.
- `landmarks` holds 21 `[x, y]` points from 0 to 1, or `null`. They are already in the mirrored view, so draw them directly over a mirrored preview: `px = x * width`, `py = y * height`. They use MediaPipe's hand order (0 is the wrist, 4 the thumb tip, 8 the index fingertip). Bones:
  `[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]]`

`progress`, sent to both players when a letter is accepted:
```json
{"type":"progress","slot":1,"done":2,"length":3}
```

`round_failed`, sent to both players:
```json
{"type":"round_failed","round":2,"reason":"timeout","slots":[2]}
```
`slots` lists the players who had not finished.

`image_pending`, then `image_ready` and `round_complete`:
```json
{"type":"image_pending","round":2,"phrase":"BIG DOG"}
{"type":"image_ready","round":2,"phrase":"BIG DOG","url":"/images/ABCD/7"}
{"type":"round_complete","round":2,"phrase":"BIG DOG","points":100,"score":200}
```
The image is a Muse-generated JPEG. If Muse is not configured, fails, or takes more than 10 seconds, it is an SVG placeholder that shows the phrase. Images last as long as the room and remain available across `play_again`, even though the new game gallery resets. Trust the HTTP content type rather than assuming JPEG from the URL. The `game_over.gallery` contains every successful round of the current game, in completion order (the example below abbreviates it to one entry).

`game_over`:
```json
{"type":"game_over","score":500,"gallery":[{"phrase":"RED CAT","url":"/images/ABCD/1"}]}
```

`room_closed`, sent to remaining connections when the reconnect grace expires (30 seconds from the first outstanding disconnect). The room is removed, images expire, and sockets close. Create a new room.
```json
{"type":"room_closed","reason":"partner_left"}
```

`error`:
```json
{"type":"error","code":"room_not_found","message":"No room with that code."}
```

| Code | Meaning | Socket |
|---|---|---|
| `join_first` | First message was not a JSON object with create, join, or rejoin (including binary first messages) | closed |
| `room_not_found` | Wrong code, or the room has closed | closed |
| `room_full` | The room already has two players | closed |
| `bad_token` | Unknown token for this room | closed |
| `bad_message` | After joining, text was invalid JSON or not a JSON object | stays open |
| `unknown_type` | After joining, unsupported/missing `type` (including repeated join messages or typed letters) | stays open |
| `frame_too_large` | Eligible binary frame exceeds 300,000 bytes; exactly 300,000 is allowed | stays open |
| `bad_frame` | Eligible frame could not be decoded or exceeds 640 pixels wide or 480 high | stays open |
| `recognizer_unavailable` | Recognition worker initialization or processing failed | closed |

Dispatch errors by `code`; `message` is human-readable. Oversized or invalid frames reset your current hold. The oversized-frame error text says "under 300000 bytes", but the implemented rejection is strictly greater than 300,000. Send JPEG as required by the contract; decoding currently uses OpenCV without a separate JPEG signature check.

## Reconnecting

When the socket closes unexpectedly, open a new one and send `rejoin` with the stored `room` and `token`. A newer connection replaces an older one for the same player. After `room_joined`, you get the current `room_state`. During `signing` you also get `round_start` and both players' `progress`, and during `game_over` you get `game_over`. Use `room_state.time_left` for the remaining signing time rather than restarting the full `round_start.time_limit`.

The grace window starts at the first disconnect and does not extend when another player disconnects. Only a valid token reclaims an occupied slot; `join_room` cannot replace a disconnected player in a full room. If grace has expired, rejoin returns `room_not_found` and closes the socket. Keep tokens private and retain them with the room code across socket replacement.

Rejoin does not replay `countdown`, `image_pending`, `image_ready`, or `round_complete` during countdown, generating, or reveal. Retain already received words/images in the UI; `room_state` does not include them or remaining countdown/reveal seconds. On `play_again`, keep room/token/slot, clear the game UI, and ready up again: both ready flags, score, round number, words, and gallery reset.

## Running the backend locally

```bash
uv sync
uv run uvicorn server.main:app --port 8000
# optional: real words and images
MODEL_API_KEY=... uv run uvicorn server.main:app --port 8000
```

Browsers allow the camera only on HTTPS or `localhost`. If the backend runs on a remote machine, tunnel it with `ssh -L 8000:localhost:8000 <server>` and open `http://localhost:8000/dev`.
