# ASL Co-op Game Backend: PRD and Handoff

Last updated 2026-10-24, during HackTX weekend. Written at the end of a requirements interview with the backend owner. Section 16 tracks build progress and is updated after every implemented task.

## How to use this document

- Read it fully before writing any code. The project folder held nothing but this file when it was written.
- Section 4 lists confirmed decisions. Do not reopen them unless the user does.
- Section 12 records the round 2 answers (confirmed 2026-10-24). The implementation plan is `docs/superpowers/plans/2026-10-24-asl-coop-backend.md`.
- Section 16 is the live build status: which tasks are done, their commits, and decisions made during the build. The code is on branch `backend-shit` of https://github.com/dabestnub/HackTX.
- Do not commit the saved ChatGPT page ("Research ASL Models.html") if it shows up in this folder. It embeds a live ChatGPT session access token.

## 1. Product

Two players cooperate in a browser game that teaches ASL fingerspelling. Each round, the server picks an adjective and a noun that form a short phrase, such as RED CAT. One player gets each word. Each player signs their word letter by letter on camera, and a classifier checks every letter. When both words are complete, the team waits while an AI image of the phrase is generated. The image is shown and the team earns points. The game ends when the team reaches 500 points.

## 2. Scope

In scope for the backend owner:
- A Python FastAPI server with WebSockets, deployed to fly.io over HTTPS.
- Server-side sign recognition. This means OpenCV, MediaPipe hand landmarks, and the pretrained classifier from https://github.com/Gj-345/asl-hand-gesture-recognition.
- Room, round, timer, and scoring logic.
- Calls to Muse (Meta Model API) for word pairs and images.
- A WebSocket protocol document for the frontend teammate.
- A minimal dev test page served by the backend, for playing the full loop with a real camera.

Out of scope: the real frontend UI (built by a teammate), accounts, persistence, leaderboards, and rooms with more than two players.

## 3. Gameplay loop (confirmed)

1. A player creates a room and receives a room code.
2. A friend joins with the code.
3. Both players ready up.
4. The server gets an adjective and noun pair, and each player gets one word.
5. Each player signs their word one letter at a time. The classifier checks each letter. Wrong signs are ignored, with no penalty.
6. Each round has a 10-second limit. Both players must finish signing their whole word before it runs out, or the round ends with 0 points and a new pair starts (Q13).
7. When both words are complete, Muse generates an image of the merged phrase. Both players wait for it to load.
8. The team earns 100 points.
9. The loop repeats until the team score reaches 500. The game then shows the image gallery and offers to play again.

## 4. Confirmed decisions

| # | Decision |
|---|---|
| Q1 | **Recognition runs on the server.** The browser sends camera frames as JPEG images over a secure WebSocket. The server decodes them with OpenCV, runs MediaPipe, then runs the classifier. The user chose this option. One concern was raised: browsers grant camera access only on HTTPS or localhost under every option, and fly.io serves HTTPS for any of them, so HTTPS does not by itself favor this option. Sending frames costs more server CPU and bandwidth than sending landmarks. The user may revisit this. |
| Q2 | **Game state lives in memory on exactly one fly.io machine.** No Redis. Run `fly scale count 1`, because `fly launch` creates two machines by default and would split the players. |
| Q3 | **Each completed word pair is worth a flat 100 points.** The team wins at 500, which is five successful rounds. |
| Q4 | **Wrong signs carry no penalty.** The player keeps trying until they sign the correct letter or the time limit runs out. The limit is 10 seconds per round for the whole word (Q13). The user did not ask for a skip button, so the timer is what gets players unstuck. |
| Q5 | **Players wait for each image** before the next round starts. |
| Q6 | **Generated words never contain J or Z.** Both letters are signed with motion, which a single-frame classifier cannot see. |
| Q7 | **Words come from Muse and are validated by the server.** Muse returns an adjective and a noun as structured JSON. The server checks that each word uses only the letters A to Z, is **3 to 5** letters long (the user lowered the cap from 7 to 5 in round 2), contains no J or Z, and has not been used earlier in the game. If Muse fails or is slow, the server uses a built-in list of about 50 pairs. The next pair is fetched during the current round. Players swap between the adjective and the noun each round. |
| Q8 | **A dropped player has 30 seconds to reconnect.** The reconnect is keyed by a player token the server issues when the player joins. |
| Q9 | **A teammate builds the frontend.** The backend owner writes the WebSocket protocol document first, and the backend implements exactly that contract. |
| Q10 | **Testing the loop requires the real camera.** The model must actually see the signs. There is no mode that accepts typed letters. |
| Q11 | **Reaching 500 ends the game.** Both players see the image gallery, and play again resets the score in the same room. |
| Q12 | **The repo's license is ignored for now, by the user's choice.** The repo has no LICENSE file, so by default the author keeps all rights. This is logged as a risk in section 13. |

## 5. Architecture

```
Browser (teammate UI, or the /dev test page)
  camera -> canvas -> JPEG -> binary WebSocket message over wss://
        |
FastAPI on fly.io (one machine, in-memory rooms)
  WebSocket handler per player
    -> holds only the latest frame per player; older frames are dropped
    -> worker thread: cv2.imdecode -> cv2.flip(frame, 1) -> MediaPipe HandLandmarker
       -> 63 features -> normalize -> classifier -> letter-acceptance rule
    -> game engine (rooms, rounds, timers, score) -> broadcasts to both players
  Muse client: word pairs (text model) and images (image model)
  HTTP: /healthz, generated images, /dev test page
```

Implementation notes:
- MediaPipe and the classifier are CPU-bound. Run them off the event loop, for example with `asyncio.to_thread`. Give each player their own HandLandmarker instance and process one frame at a time per player.
- Keep the game engine free of I/O. It should take events (letter accepted, timer fired, image ready) and return messages to send. That keeps it testable with a stub recognizer.
- The frontend must send raw, unmirrored camera frames. It may mirror the on-screen preview only. The server flips frames to match how the repo's app runs (section 6).

## 6. Recognition facts (from repo commit 9de5b22)

Files needed from the repo:
- `ml/models/best_model.pth` (326 KB). This is a state_dict saved with tensors on `cuda:0`, so load it with `map_location="cpu"`.
- `ml/hand_landmarker.task` (7.8 MB). This is the MediaPipe Tasks hand model.
- `ml/models/classes.pkl` is a plain list `['A', ..., 'Z']`, all 26 letters in order, saved with joblib. Hardcode this list instead of unpickling the file.

Clone the repo again in the next session. The earlier clone lived in a temporary session folder.

Model (`ml/model.py`, class `GestureModel`): a single `nn.Sequential` with about 78k parameters.
```
Linear(63,128) BN(128) ReLU Dropout(0.3)
Linear(128,256) BN(256) ReLU Dropout(0.3)
Linear(256,128) BN(128) ReLU Dropout(0.2)
Linear(128,26)
```
The state_dict keys are `network.{0,1,4,5,8,9,12}.*`. The output covers 26 classes, including J and Z, with no space or nothing class.

Preprocessing is identical in `app.py:48-66` and `convert.py:25-47`. Reproduce it exactly:
```python
hand = result.hand_landmarks[0]            # first hand only, handedness ignored
for lm in hand: landmarks.extend([lm.x, lm.y, lm.z])   # x0,y0,z0,...,x20,y20,z20
landmarks = np.array(landmarks, dtype=np.float32)
landmarks = landmarks - landmarks.min()    # one global min over all 63 values
if landmarks.max() != 0: landmarks = landmarks / landmarks.max()
```

How the repo's live app (`app.py`) behaves. The user ran it and confirmed it works, so match it exactly:
- It mirrors every webcam frame with `cv2.flip(frame, 1)` before MediaPipe (`app.py:105`). The training images were not flipped.
- It uses the MediaPipe Tasks `vision.HandLandmarker` in `RunningMode.IMAGE`, not the legacy `solutions.hands` API.
- HandLandmarker settings: `num_hands=1`, `min_hand_detection_confidence=0.6`, `min_hand_presence_confidence=0.6`, `min_tracking_confidence=0.5`.
- It takes the argmax of the softmax. No confidence threshold gates acceptance.
- It accepts a letter after about 21 identical predictions in a row (`STABLE_THRESHOLD = 20`). Frames with no hand detected do not reset the count. Holding a sign appends it again every 20 frames.
- requirements.txt pins no versions.

Reported accuracy is 98.97% on a held-out test split built from 20-frame chunks of what is probably the Kaggle ASL Alphabet dataset. It has not been validated on new signers.

## 7. Muse (Meta Model API) facts

Docs: https://dev.meta.ai/docs. Keys are created in the dashboard under API keys and are shown only once. Store the key as the fly secret `MODEL_API_KEY`.

- **Base URL and auth.** The base URL is `https://api.meta.ai/v1`, with the header `Authorization: Bearer $MODEL_API_KEY`.
- **Client library.** There is no Meta SDK. The OpenAI Python client works, but the backend uses plain `httpx` POSTs: two calls, Meta-only fields, and httpx is already needed for tests.
- **Text models.** `muse-spark-1.3`. Avoid the `-contributor` variants, because they let Meta train on prompts and outputs.
- **Structured output.** Chat Completions accepts `response_format={"type": "json_schema", "json_schema": {...}}`, and `strict` defaults to false. Pydantic parsing through `client.beta.chat.completions.parse()` works.
- **Images.** `POST /v1/images/generations` with `model="muse-image-1.0"`. It takes `prompt`, `n`, and `size` as a `"WxH"` string that sets the aspect ratio only. `response_format` is `b64_json` (the default) or `url`, which returns a temporary signed URL with no stated expiry. `output_format` is webp (the default), png, or jpeg. Meta adds `reasoning_strength` ("high" by default, "low" is faster) and `tool_enablement` (web and image search, on by default). Unknown fields such as `seed` return HTTP 400. Verified 2026-10-24 against `/docs/api-reference/images/schemas`: `reasoning_strength` is a top-level enum `low|high`; `tool_enablement` is an object with optional booleans `enable_image_search`, `enable_web_search`, `enable_shell`; the image is at `data[0].b64_json`. Structured output: `response_format` json_schema with `strict: true` needs `additionalProperties: false` and every property listed in `required`.
- **Pricing and limits.** Muse Spark costs $1.25 per million input tokens and $4.25 per million output tokens, limited to 3,000 requests per minute. Images cost $0.01 each, limited to 150 requests per minute. Image latency is not documented. No free tier is confirmed in Meta's docs.

Word validation regex, with J and Z excluded: `^[A-IK-Y]{3,5}$`.

## 8. Game state machine (proposed)

```
LOBBY (0-2 players, ready flags)
  -> both ready -> COUNTDOWN (3 s, assumption)
  -> SIGNING (each player has a word, a letter index, and a timer)
       both words complete -> GENERATING_IMAGE (players wait, Q5)
                           -> REVEAL (score += 100)
                           -> score >= 500 ? GAME_OVER : COUNTDOWN
       10 s round limit hit -> ROUND_FAILED (0 points) -> COUNTDOWN with a new pair (Q13)
GAME_OVER -> play again -> LOBBY or COUNTDOWN with score reset
A disconnect pauses the room for 30 s (Q8). If the player does not return, the room ends.
```

## 9. WebSocket protocol draft

This is a draft. Finalize it in `docs/protocol.md` once section 12 is settled, and share it with the frontend teammate. Messages are JSON text frames, except camera frames, which are binary.

Client to server:
- `{"type":"create_room","name":"..."}`
- `{"type":"join_room","room":"ABCD","name":"..."}`
- `{"type":"rejoin","room":"ABCD","token":"..."}`
- `{"type":"ready","ready":true}`
- A binary frame containing one JPEG camera image. It is processed only during SIGNING.
- `{"type":"play_again"}`

Server to client:
- `room_joined`: room code, your slot (1 or 2), player token.
- `room_state`: phase, players (name, ready, connected), score, target 500.
- `round_start`: round number, both words, your word, timer info.
- `recognition`: sent to the signer only. Includes the current guess, its confidence, hold progress, and hand landmarks for an overlay. The exact contents depend on Q17.
- `progress`: sent to both players. Includes the slot, letters completed, and word length.
- `round_failed`: reason `timeout`, plus which slot timed out.
- `round_complete`: the phrase, points awarded, and the new score.
- `image_pending`, then `image_ready` with an image URL and the phrase.
- `game_over`: the final score and the gallery as a list of phrase and URL pairs.
- `error`: a code and a message.

## 10. Deployment

- Use a Dockerfile based on `python:3.12-slim`. MediaPipe wheels for Python 3.13 are unverified.
- MediaPipe's pip dependency `opencv-contrib-python` probably needs `libgl1` and `libglib2.0-0` from apt. Verify this.
- fly.io settings: one machine, a Dallas region (`dfw`), `auto_stop_machines = "off"`, and `min_machines_running = 1`, so in-memory rooms survive during the event. A machine of shared-cpu-2x with 2 GB is proposed (Q21).
- fly.io terminates HTTPS, so the WebSocket runs over `wss://` on the same host.
- Secrets: `MODEL_API_KEY`.
- flyctl is not installed on the dev server yet.

## 11. Testing

- **Camera loop (required, Q10).** Play the full game with real signing through the /dev page or the teammate's UI. Two players can use two browser tabs on one laptop sharing a webcam, or two devices.
- **Secure origin.** The dev server is a remote Linux machine with no webcam. A browser reaching it over plain HTTP gets no camera access. Use `ssh -L 8000:localhost:8000 <server>` so the browser sees `localhost`, or test against the fly.io HTTPS URL.
- **Automated tests.** Cover the game engine (rounds, timers, scoring, reconnect, J and Z exclusion, word validation) with a stub recognizer injected in tests only. This is not a production bypass. Add one parity test comparing the server's classifier output with the original torch model on a few inputs. Details depend on Q20.
- **No Muse key.** Without a key, use the fallback word list and a placeholder image, so local runs cost nothing. Details depend on Q19.

## 12. Round 2 answers (confirmed 2026-10-24)

| # | Decision |
|---|---|
| Q13 | **One 10-second limit per round, for the whole word.** Both players must finish their word before it runs out. A timeout gives 0 points and starts a fresh pair. Words are 3 to 5 letters, so 5 holds of 0.6 s fit. |
| Q14 | **A letter is accepted when the expected letter is the top prediction with confidence of at least 0.8, held for 0.6 s.** A frame with no hand, a different guess, or lower confidence resets the hold. J and Z are masked out of the prediction. |
| Q15 | **Repeated letters** (BOOK) are accepted by holding through a second full hold. There is no need to lower the hand. |
| Q16 | **Frames are raw, unmirrored JPEGs of at most 640x480, at most 10 per second.** The server keeps only the newest frame per player and flips it as `app.py` does. |
| Q17 | **The signer gets their guess, confidence, hold progress, and 21 landmark points** for each processed frame. Both players see each other's letter progress. |
| Q18 | **The classifier runs in numpy** with BatchNorm folded into the linear layers. Torch is used only by the one-time export script, which also writes the parity fixture the tests use. |
| Q19 | **Images come back as base64**, are held in room memory, and are served at a URL announced over the WebSocket. If generation fails or takes over **10 seconds**, a placeholder showing the phrase is used. Requests use `reasoning_strength="low"` with web and image search off. |
| Q20 | **The backend serves a minimal test page at /dev.** Automated tests cover game logic, plus a protocol smoke test. A stub recognizer exists only inside the tests. |
| Q21 | **One shared-cpu-2x machine with 2 GB in dfw**, auto-stop off, Python 3.12 image. |
| Q22 | **The user is unsure where the UI will be hosted.** This does not block anything. Browsers do not apply CORS to WebSockets, and `<img>` tags load images from any origin. The backend still allows all origins on HTTP so `fetch()` works from anywhere. A UI can be hosted on Vercel, Netlify, or locally. |

Other facts from round 2: the user has run the repo's app and it works. They have a Muse key and will supply it later, set as the `MODEL_API_KEY` environment variable or fly secret.

## 13. Risks

- **Model rights.** The repo has no license, so by default the author keeps all rights to its code and weights. The user chose to ignore this for now. Asking the author, or retraining on the public dataset, are the remedies.
- **New signers.** The user ran the repo's app and it works for them. Accuracy for other signers and lighting is still unmeasured, and the 0.8 confidence bar may need tuning after the first camera test.
- **Server CPU.** Running MediaPipe on fly.io for two players at 10 frames per second each needs measuring. Frame dropping keeps latency bounded.
- **Image waits.** Players wait for every image, and image latency is undocumented. The 10-second placeholder fallback in Q19 limits the damage.
- **Muse cost.** Muse needs a paid key with no confirmed free tier. Costs are tiny at one cent per image.
- **Timer pressure.** 10 seconds for a 5-letter word leaves 7 seconds of slack beyond the 3 seconds of holds. Tune `TIME_LIMIT` after the first camera test if needed.

## 14. Build order

Superseded by the 13-task plan in `docs/superpowers/plans/2026-10-24-asl-coop-backend.md`. Progress is in section 16.

## 15. Environment facts (dev server)

- The dev server is a remote x86_64 machine running Ubuntu 22.04, with 4 NVIDIA GPUs, no webcam, and no `/dev/video*`.
- Its Python is 3.13.9 from miniconda. cv2 4.13 and numpy 2.3 are installed. fastapi, uvicorn, websockets, torch, and mediapipe are not.
- `uv` 0.13.4 and `pip` are available. Docker is installed and the daemon is reachable. flyctl is not installed.
- The project uses a uv-managed CPython 3.12.15. mediapipe 1.1.1 installs on it and brings opencv-contrib-python 5.0; do not add another OpenCV wheel. A blank 640x480 frame takes about 17 ms in HandLandmarker on this CPU.
- Git: commits are authored as `kevnlin`, and pushes use `~/.ssh/id_ed25519_kevnlin` (set in the repo's `core.sshCommand`). The machine's default key and global identity belong to someone else.

## 16. Implementation status

Updated after every implemented task. Each task is built by one coding agent, then passes a spec review and a code quality review before it is pushed to `backend-shit`.

| Task | What | Status | Commit |
|---|---|---|---|
| 1 | Scaffold: uv, Python 3.12, dependencies, hand landmarker model | Done | d52c4cc |
| 2 | Numpy classifier, one-time torch export, parity test | Done | a829548 |
| 3 | Letter acceptance: 0.8 confidence held 0.6 s, J and Z masked | Done | ea78f83 |
| 4 | Word validation and fallback words | Done | ff3947a |
| 5 | MediaPipe recognizer matching the repo's `app.py` | Done | 82725ea |
| 6 | Engine 1/3: rooms, lobby, countdown, round start | Done | 8b5710a |
| 7 | Engine 2/3: signing, images, scoring, game over | Done | 72304ba |
| 8 | Engine 3/3: timeout, disconnect, reconnect | Done | 9b9414d |
| 9 | Muse client and placeholder image | Not started | |
| 10 | FastAPI server and WebSocket protocol | Not started | |
| 11 | `docs/protocol.md` for the frontend teammate | Not started | |
| 12 | `/dev` camera test page | Not started | |
| 13 | Dockerfile and fly.toml | Not started | |

Tests: 47 passing at Task 8 (`uv run pytest -q`).

Decisions made during the build, beyond the interview answers:
- After an image appears, it shows for 5 seconds (`reveal`) before the next 3-second countdown. Both values are in `server/config.py`.
- J and Z are masked by zeroing their probabilities. The confidence compared with 0.8 is the raw softmax value, not renormalized.
- The fallback lists hold 29 adjectives and 40 nouns. If a game uses every fallback word (29 or more rounds, possible with many timeouts), words repeat instead of the game stopping.
- `words.normalize` returns an empty string for anything that is not a string, so a null word from Muse is rejected instead of becoming "NONE".
- Frames over 640x480 pixels are refused before decoding (`OPENCV_IO_MAX_IMAGE_PIXELS`), so a small JPEG that claims a huge size cannot exhaust memory.
- A third player joining a full room is refused by the server (Task 10), not by the engine.
- Known edge case: if frames stop mid-hold and resume on the same letter, the hold can complete at once. This is low risk; revisit it if the camera test shows false accepts.
- If an image arrives while a player is reconnecting, its reveal runs longer than 5 seconds by the length of the pause. This is harmless and left as is.
- Not tested yet: a frame with a real hand. The first camera test is the first check of that path.
