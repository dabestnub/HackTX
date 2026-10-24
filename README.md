# SignSync — Multiplayer ASL Game Powered by AI

SignSync is a collaborative two-player game where players alternate signing ASL fingerspelling letters, complete words together, generate shared AI imagery, and earn points against a two-minute timer.

## UI reference samples

The supplied UI samples are stored in [`ui-samples/`](ui-samples/).

![SignSync UI reference 1](ui-samples/signsync-ui-reference-1.png)

![SignSync UI reference 2](ui-samples/signsync-ui-reference-2.png)

## Current website prototype

The static website UI is in [`index.html`](index.html) and can be hosted by a static site host that serves `index.html` from the site root. For local testing, run `python3 -m http.server 8000` from this folder and open `http://localhost:8000`. Camera access requires HTTPS on a hosted site (localhost is allowed for local testing).

The visual refresh applies the [UI UX Pro Max guidance](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill): semantic design tokens, stronger type and content hierarchy, restrained surfaces and shadows, responsive reflow, visible keyboard focus, and reduced-motion support. The skill is design guidance only; the site does not depend on its runtime or tooling.

Selecting **Enable local ASL recognition** asks the browser for camera permission. The page uses MediaPipe Tasks Vision and ONNX Runtime Web to process camera frames and landmarks in the browser; it does not upload or store them. Runtime code/WASM and the MediaPipe hand-landmarker asset are loaded from jsDelivr and Google's model hosting, so the browser needs network access for those downloads; they do not receive camera frames. Predictions are shown with the upstream model's raw softmax score and require explicit confirmation. The app asks for a hand release before arming a subsequent letter. A live browser test predicted and accepted C. J/Z motion recognition is not verified.

This remains a single-browser practice prototype: turns alternate locally, the second player is not connected, and the timer, authoritative scoring, multiplayer backend, and AI image generation are not implemented. The simulation control is clearly labeled for testing.

To regenerate the browser classifier assets from a local checkout of the required ASL repository, install its requirements plus ONNX in a Python environment and run:

```bash
python -m pip install -r /path/to/asl-hand-gesture-recognition/requirements.txt onnx
python scripts/export_asl_model.py --source /path/to/asl-hand-gesture-recognition
```

The export script creates `assets/asl_classifier.onnx` and `assets/asl_classes.json`.

## Required ASL recognition repository

Use and inspect this existing repository before implementing a new model:

https://github.com/Gj-345/asl-hand-gesture-recognition

Reuse its existing recognition pipeline where feasible. Verify dependencies, trained weights, preprocessing, supported letters, and webcam inference directly from the source. Do not claim real recognition works until it has been tested.

### Recognition repository source review

The required repository's source has been inspected. It contains trained classifier weights, a class-label artifact, and the MediaPipe hand landmarker asset. Its `app.py` expects to be run from the recognition repository root with `python app.py`; it captures camera device 0 through OpenCV, extracts 21 hand landmarks (63 x/y/z values), normalizes them, classifies them using PyTorch, and reports a softmax score. Its `requirements.txt` lists unpinned dependencies.

The model and MediaPipe detector were loaded in an isolated Python 3.12 environment. A live webcam test recognized one C handshape: C was the prediction at the 21st consecutive frame with a raw softmax score of 0.9990. The frame was processed locally and not saved. The exported ONNX model matched PyTorch on eight test vectors (maximum logit absolute difference 5.72e-6, with no predicted-class mismatches), and ONNX Runtime Web loaded it in a browser and returned 26 logits. The static website was then tested with a live C handshape: the browser-local pipeline produced a stable C prediction, confirmation appended it to the word, and the hand-release re-arm worked. This verifies one static letter, not general accuracy or all classes. The upstream README lists A-Z classes, but the classifier processes one frame of static landmarks, so support for motion-dependent J and Z is not verified. Its current 20-frame append behavior also does not require a neutral pose before accepting another letter. A raw softmax score is not independently calibrated confidence.

The user confirmed permission to bundle the classifier. The static website includes a generated ONNX classifier asset. No LICENSE file was found in the inspected upstream checkout, so obtain and retain permission details if distributing this project beyond the user's authorized use. The MediaPipe hand-landmarker model and runtime are loaded from their published hosting locations, not copied into this project.

## Recommended first move

Ask your coding agent to inspect the recognition repository and get real webcam inference working before building the rest of the app.

Suggested first instruction:

> Inspect https://github.com/Gj-345/asl-hand-gesture-recognition. Read the README and source files, identify the model weights, dependencies, preprocessing, label mapping, and webcam inference entry point. Give me the exact setup/run commands and test real webcam inference if possible. Do not build the rest of SignSync yet. Report missing files, unsupported letters, and compatibility issues.

## Step-by-step development order

1. **Validate recognition locally.** Follow the repository's documented setup, run its webcam demo, test several letters, and investigate J/Z support.
2. **Choose an integration path.** The current prototype uses browser-local MediaPipe and ONNX Runtime Web inference because classifier export was verified against the upstream PyTorch model. Keep camera frames local. Use a Python inference service only if browser conversion becomes unreliable and after explicit consent for any frame transfer.
3. **Build a multiplayer vertical slice.** Connect two separate browsers, show camera previews, alternate turns, and synchronize confirmed letters.
4. **Add words, score, and timer.** Use a server-authoritative two-minute timer and team score target of 500. Prevent duplicate scoring.
5. **Add image generation.** Verify the actual Muse Image API/service access and implement an adapter with loading/error states. Clearly label any mock fallback.
6. **Add communication.** Implement real-time text chat and optional speech-to-text.
7. **Add results and sharing.** Show statistics and use a supported, user-initiated Facebook sharing flow where available.
8. **Polish and test.** Test two devices, low-confidence predictions, camera errors, disconnects, generation failures, mobile layouts, and accessibility.

## Core gameplay

- Exactly two players join a shared room.
- Players alternate signing individual ASL fingerspelling letters.
- Accepted letters appear for both players.
- A completed word triggers one image-generation request.
- Both players see the same generated image.
- The team earns points toward 500 within two minutes.
- Text chat is available; voice transcription is optional.
- Results can be shared only after an explicit user action.

## Architecture notes

- **Frontend:** React or Next.js, TypeScript, responsive CSS, browser webcam APIs.
- **Multiplayer:** WebSockets/Socket.IO or another real-time service.
- **Recognition:** Python, MediaPipe, PyTorch, and the existing repository's model, subject to source verification.
- **Backend:** authoritative turns, word validation, scoring, timer, room membership, and idempotent game events.
- **Image generation:** a modular adapter for the verified Muse Image integration.
- **Persistence:** store only necessary session and score data; do not persist webcam footage by default.
- **Security:** keep credentials server-side and validate every client event.

## External service caveats

Muse Image access, API endpoints, and Facebook publishing permissions must be verified in the actual development environment. Do not invent endpoints or assume direct timeline publishing is allowed. Provide a clearly labeled fallback when a service is unavailable.

---

## Full build prompt and guide

The sections below preserve the full build brief and detailed implementation checklist from the project planning document.

SignSync — AI build prompt and guide
Page 1
SignSync
Full AI Build Prompt + Step-by-Step Hackathon Guide
Project concept: A two-player multiplayer game where people alternate signing ASL fingerspelling letters, complete a word
together, generate a shared AI image, and earn points against a two-minute timer.
Required recognition repository: https://github.com/Gj-345/asl-hand-gesture-recognition
Reference design: Attach the two concept images when you give the prompt to your coding agent.
How to use this PDF
1
Follow the practical step-by-step guide near the beginning of this document.

2
Start with repository inspection and a real webcam recognition test before building the entire app.

3
Copy the full build prompt in Part 2 into your AI coding tool, and attach both design reference images.

4
Build and test one phase at a time. Do not treat mock recognition or mock image generation as a completed integration.

Contents
●
Part 1 — Step-by-step instructions after the recommended first move

●
Part 2 — Complete prompt to give your AI coding agent

●
Part 3 — Practical definition of done and demo checklist

SignSync — AI build prompt and guide
Page 2
Part 1 — Step-by-Step Instructions
Recommended first move: ask your coding agent to inspect the GitHub recognition repository and verify that webcam
recognition works before building the full application. This reduces the largest technical risk early.
Step 1: Open your coding environment
1
Open the AI coding tool you plan to use (for example, an IDE coding agent or repository-aware app builder).

2
Create or open a project folder for SignSync. If you already have a starter repository, use that rather than creating a
second project.
3
Attach the two visual reference images to the agent so it understands the intended layout and style.

4
Give the agent the GitHub URL: https://github.com/Gj-345/asl-hand-gesture-recognition.

Step 2: Ask the agent to inspect the recognition repository first
Paste this short instruction before the full build prompt:
Before implementing the app, inspect https://github.com/Gj-345/asl-hand-gesture-recognition. Read its
README and source files, identify the model weights, dependencies, preprocessing, label mapping, and
webcam inference entry point. Tell me the exact commands to install and run it, then test real webcam
inference if the environment permits. Do not build the rest of SignSync yet. Report any missing model
files, unsupported letters, or compatibility problems.
Step 3: Verify recognition locally
1
Follow the repository's own README and use its documented installation and run commands. Do not guess dependency
versions if the repository specifies them.
2
Create a Python virtual environment if the project recommends one, then install the listed dependencies.

3
Run the repository's provided inference or webcam demo.

4
Test several letters under the lighting and camera conditions you expect during the hackathon.

5
Check how the program returns its predicted class and confidence, and whether it supports motion-dependent letters such
as J and Z.
6
Record the exact command that works and any errors. Ask the coding agent to fix environment issues before moving on.

Success checkpoint: You can run the existing recognition pipeline and observe predictions from a real webcam, or you
have a clear, documented blocker and a specific fallback plan.
Step 4: Choose the recognition integration path
●
Recommended starting point: Keep the model in Python and wrap its inference logic in a small FastAPI service if that is
compatible with the repository.
●
Have the browser capture webcam frames and send only the minimum necessary data for inference. Avoid uploading or
storing continuous video unless essential.
●
If local browser inference is feasible, compare it against the original model output before choosing it.

●
Define a simple response contract such as predicted letter, confidence (if available), and recognition status. Do not invent
confidence values if the model does not provide them.
Step 5: Build the smallest multiplayer vertical slice
1
Create a room that two separate browsers can join.

2
Show each player's nickname, connection state, and webcam preview.

3
Implement turn ownership and a shared letter display.

SignSync — AI build prompt and guide
Page 3
4
Connect one recognition result to a confirm/retry control.

5
When a player confirms a letter, have the backend validate the turn and broadcast the accepted letter to both players.

6
Add a neutral-pose or release step so holding one hand shape does not repeatedly submit the same letter.

Success checkpoint: Two different browsers can take turns and both see the same accepted letters without refreshing the
page.
Step 6: Add words, scoring, and timer
1
Start with a small list of words and a clearly defined target word for each round.

2
Use a server-authoritative two-minute timer.

3
Implement the shared team score and progress bar toward 500 points.

4
Award points only on validated events, and give each event a unique identifier to prevent duplicate scoring.

5
End the round when the score target is reached or time runs out.

6
Test reconnects and simultaneous messages so they cannot reset the timer or duplicate points.

Step 7: Add Muse Image generation
1
Verify the exact Meta Muse Image product/API access, endpoint, credentials, and terms available to your team.

2
Implement image generation behind a separate backend adapter so the game does not depend on a specific vendor
implementation.
3
Trigger generation only after a valid word is completed.

4
Show a loading state, then broadcast the same generated image reference to both players.

5
Add timeout, error, retry, and clearly labeled mock-mode behavior if the real service is unavailable.

Step 8: Add chat and optional voice transcription
1
Implement real-time text chat between the two players.

2
Add an optional microphone control for speech-to-text.

3
Request microphone permission only when the user activates the feature.

4
Let the player review or edit the transcript before sending.

5
Ensure that the game remains fully playable without speech or a microphone.

Step 9: Add results and Facebook sharing
1
Create a results screen with score, words, accepted letters, images generated, and time taken.

2
Verify the supported Facebook sharing flow and permissions for your app.

3
Do not assume direct timeline publishing is permitted. Use a supported share dialog, shareable card, or copy-link fallback.

4
Require an explicit user action before sharing, and exclude webcam footage and private chat.

Step 10: Polish and test the demo
●
Match the references: warm yellow background, rounded gray-blue player panels, blue progress bar, friendly illustrations,
dotted texture, and clear typography.
●
Test in two separate browsers or devices, not only two tabs sharing the same camera.

●
Test camera denial, low confidence, hand loss, disconnects, reconnects, image-generation failure, and timer expiry.

●
Check the layout at desktop and mobile sizes.

●
Write a README with setup instructions, environment variables, services used, and known limitations.

SignSync — AI build prompt and guide
Page 4
Suggested order for a hackathon
1) Recognition works locally → 2) Two-player room → 3) Shared accepted letters → 4) Timer and scoring → 5) Real image
generation → 6) Chat and voice → 7) Sharing and polish.
If time gets tight, prioritize a reliable end-to-end demo of the core game loop over adding many partially
working features.

SignSync — AI build prompt and guide
Page 5
Part 2 — Complete Prompt for the AI Coding Agent
Copy everything in this part into your AI coding agent. Attach the two reference images and provide the GitHub repository
link.
Build SignSync: A Multiplayer ASL Game Powered by AI
Project Overview
Build a functional, polished, multiplayer web application called SignSync that brings people closer together through
American Sign Language (ASL), AI-generated imagery, and real-time communication.
The application connects two players in a shared game room. Players take turns signing individual ASL fingerspelling letters
using their webcams. The system recognizes each letter, combines the letters into a word, and generates an AI image
representing that word using Meta's Muse Image generation model. Both players see the same letters, image, timer, score,
and game progress in real time.
For example, if the target word is CAT: Player 1 signs C; Player 2 signs A; Player 1 signs T. The application recognizes and
displays C → A → T. Once the word is completed correctly, Meta Muse Image generates a visual representation of a cat.
Both players see the generated image, and the team earns points toward 500 within two minutes.
The core objective is to demonstrate how real ASL fingerspelling recognition, generative AI, and multiplayer communication
can help two people connect through a shared experience.
1. Required Technology: Existing ASL Recognition Repository
Use this specific GitHub repository as the foundation for recognizing ASL fingerspelling letters:
https://github.com/Gj-345/asl-hand-gesture-recognition.
Do not build an entirely new ASL recognition model from scratch. First inspect and reuse the existing repository.
The repository is intended to provide hand-gesture recognition using MediaPipe and a PyTorch classification pipeline. Verify
the actual implementation, model files, preprocessing, class labels, and inference requirements directly from the repository
before integrating it.
Recognition integration requirements:
 Inspect the README, source code, dependencies, trained weights, and inference workflow.
 Determine how webcam frames are processed and how hand landmarks become model inputs.
 Preserve the existing architecture, preprocessing, label mapping, and trained weights.
 Adapt the recognition pipeline for a browser application; do not assume a desktop interface runs directly in a browser.
 Prefer a Python inference service using FastAPI or a similar framework if practical.
 Alternatively, use browser inference only if the model can be converted and deployed reliably without changing its expected
inputs or predictions.
 Keep recognition modular and distinguish actual inference from mock predictions.
Recognition workflow: the player grants webcam permission; frames are captured; the pipeline detects the hand and predicts
a letter; a stability filter checks reliability; the player confirms or retries; the backend validates the turn; the accepted letter
appears for both players; and the turn passes to the other player.
Recognition quality: use configurable confidence thresholds when available, stable predictions across frames, a
neutral-pose/release re-arming mechanism, confirmation or correction controls, retry support, hand-not-detected feedback,
and duplicate-event prevention. Do not invent confidence values if the model does not provide them.
The repository is intended for fingerspelling recognition, not general ASL sentence translation. Verify support for every
required letter, especially motion-dependent J and Z. Do not claim a letter works until tested.

SignSync — AI build prompt and guide
Page 6
2. Multiplayer Lobby and Player Connection
Build a lobby that connects exactly two players per game session. Support room creation, joining by code or invitation link,
nicknames, connection/readiness status, webcam permission, camera preview, ready buttons, synchronized start,
reconnection feedback, and graceful leaving.
Use WebSockets, Socket.IO, Supabase Realtime, or another suitable real-time service. The backend must be authoritative
for turns, scoring, accepted letters, and round completion. Prevent simultaneous submissions from corrupting game state.
3. Core ASL Gameplay: Alternate Letters
The target word is CAT. Player 1 signs and confirms C; Player 2 signs and confirms A; Player 1 signs and confirms T.
Accepted letters appear in a shared display, and the turn alternates after each accepted letter. When the word is completed
correctly, trigger image generation.
Display the target word or challenge, accepted letters, next expected letter when appropriate, active player, webcam preview,
last prediction, confidence/reliability indicator, confirm/retry controls, team score, and progress bar.
For the default mode, let both players see the target word. Optionally add a harder clue-based mode. Include a small
difficulty-based word bank: easy CAT, DOG, SUN, HAT; medium HOME, LOVE, BOOK, SMILE; harder FRIEND, FAMILY,
PEOPLE, CONNECT.
Validate completed words on the backend. Do not award completion points for arbitrary strings or repeated submissions.
Show how many letters remain.
4. Meta Muse Image Generation
Use Meta's Muse Image generation model to create an image representing each successfully completed word. For CAT,
generate a playful cat image; for HOME, a cozy house; for LOVE, a warm imaginative representation of love; for FRIEND,
two people connecting; for SUN, a bright sun in a playful scene.
Verify the actual API/service availability, authentication, endpoint, and request format. Do not invent endpoints or assume
access. If unavailable, create a modular image-generation adapter with a clearly labeled mock mode or configurable
alternative.
Keep generation asynchronous. Show a loading state, broadcast the same image to both players, cache results where
appropriate, prevent duplicate requests, and handle timeouts, rate limits, failures, and retries. The image should not block
gameplay controls.
5. Two-Minute Timer, Team Score, and Progress Bar
Each round lasts two minutes and the team aims to reach 500 points. Use a prominent countdown and server-authoritative
timestamps so both clients remain synchronized. End the round when time expires or the target is reached.
Suggested initial scoring: correct accepted letter = 10 points; completed word = 50 bonus points; successful image
generation = 25 bonus points; optional streak bonus = up to 25 points. Keep values configurable. The score is shared. Do not
deduct points for recognition errors by default.
Create a progress bar from zero toward 500, display the numeric score (for example, 230 / 500), animate increases, and
celebrate milestones. Ensure each letter, word, and image event awards points at most once.
At the end, show final team score, words completed, accepted letters, images generated, time taken to reach the target if
achieved, longest correct-letter streak, play-again, and share-results controls.
6. Real-Time Text Chat
Include a chat panel where players can send and receive text messages in real time. Display sender nickname and
timestamp, update without refresh, validate messages, apply sensible length limits, and keep chat accessible during
gameplay. Chat must work even if the camera is disabled or recognition is unavailable.

SignSync — AI build prompt and guide
Page 7
7. Optional Voice Transcription
Add an optional microphone control that converts spoken language into text using a supported browser speech-recognition
API or suitable transcription service. Request microphone permission only after activation, display the transcript, allow editing
before sending, show when transcription is active, and handle unsupported browsers and permission failures. Stop recording
when disabled or when leaving the room. Voice transcription must not be required to play.
8. Facebook Sharing and Graph API
Allow players to share a results card with team score, words completed, accepted letters, generated images, time taken, a
short message about connecting through ASL and AI, and an optional generated image.
Verify supported Facebook sharing mechanisms and permissions. Do not assume direct personal-timeline publishing is
allowed. If it is not supported, use a supported sharing dialog, shareable results card, or copy-link option. Sharing must be
optional and user-initiated. Never share webcam footage, private chat, tokens, or sensitive data. If app review or permissions
are unavailable, implement a working fallback and document the limitation.
9. UI Design and Visual Direction
Use the two attached reference images as design inspiration. The style should be playful, warm, collaborative, and
memorable.
Palette: warm pastel yellow background; light gray-blue player panels; blue progress bar; red/coral accent; occasional purple
highlights; dark brown or black readable text. Use rounded cards, friendly hand-drawn illustrations, soft shadows, subtle
animations, dotted texture, large letter displays, and approachable copy.
Gameplay layout: top section for room status, timer, score, and progress bar; main section for two player panels with
webcam previews and active-turn indicators; shared word display; prominent generated-image area with loading state and
completed-word caption; bottom chat, microphone/transcription toggle, and game controls.
On mobile, stack player panels vertically and keep letters, score, and generated image easy to see. Do not replace
functioning webcam previews with static illustrations.
10. Accessibility and Inclusion
Provide visual turn and status indicators, text feedback, keyboard-accessible controls, sufficient contrast,
camera/microphone permission explanations, optional transcription, no audio-only instructions, friendly recognition-error
states, camera-off/reconnecting states, and hand-positioning instructions.
Explain that automated recognition may be inaccurate. Do not imply that fingerspelling recognition understands ASL
grammar, facial expressions, or full signed conversations. Do not assume all Deaf or hard-of-hearing people communicate in
the same way.
11. Recommended Architecture
Frontend: React or Next.js, TypeScript, Tailwind CSS or equivalent, browser webcam APIs, and WebSockets/Socket.IO. It
renders the interface, captures or locally processes webcam frames, displays predictions, handles confirmation/retry, and
renders the shared game state.
Multiplayer backend: Node.js with Express and Socket.IO, or equivalent. It manages rooms, connections, turns, target words,
accepted letters, authoritative scores/timers, broadcasts, reconnections, image-generation coordination, and final statistics.
ASL recognition service: Python, FastAPI if suitable, MediaPipe, PyTorch, and the existing repository's trained model. It
detects hands, extracts landmarks, runs the classifier, and returns predicted letters and confidence information where
supported.
Image-generation service: a dedicated adapter for the verified Muse Image integration. It accepts completed words, builds
prompts, submits generation requests, returns image references, and handles asynchronous generation and failures.
Database: PostgreSQL or equivalent for necessary user/guest identifiers, session metadata, completed words, final scores,
statistics, and image references. Use Redis/shared state if needed for multiple backend instances. Do not persist webcam
video or private chat by default.

SignSync — AI build prompt and guide
Page 8
Security: keep credentials on the backend; validate client events; verify room membership; apply request limits; do not trust
browser-submitted scores or turn changes; use secure WebSockets in production; and obtain consent for camera,
microphone, and external sharing.
Process webcam data locally when possible. If server inference is required, minimize transmitted data and avoid retaining
frames.
12. Real-Time Game Event Design
Suggested client-to-server events: room:create, room:join, player:ready, letter:confirm, letter:retry, chat:send,
voice:transcript, game:restart, game:leave.
Suggested server-to-client events: room:updated, player:connected, player:disconnected, turn:changed, letter:recognized,
letter:accepted, word:completed, image:generation_started, image:generation_completed, image:generation_failed,
score:updated, timer:updated, chat:received, game:completed.
These names are suggestions, not mandatory APIs. Use consistent payload schemas and validate every event. Give
accepted actions unique event or idempotency identifiers so retries cannot award duplicate points. The backend determines
whether an accepted letter is valid for the player, turn, word, and round.
13. Implementation Plan
Phase 1 — Inspect and validate the recognition repository: read documentation; install dependencies; verify inference;
confirm inputs and output labels; test a real webcam; identify unreliable letters; document integration.
Phase 2 — Build the core multiplayer game: room creation/joining; connect two players; webcam previews; real ASL
integration; alternating turns; confirmation/retry; synchronized letters; word completion; timer; scoring. If necessary, use a
clearly labeled development mock while preserving the integration interface.
Phase 3 — Add AI image generation: verify Muse access; generate after a valid word; synchronize the image; add loading,
timeout, failure, and retry states.
Phase 4 — Add communication: real-time text chat; optional voice transcription; permission handling and accessibility
improvements.
Phase 5 — Add results and sharing: results screen; shareable results card; supported Facebook sharing; document
permissions and limitations.
Phase 6 — Polish and test: visual refinements; webcam/browser/lighting tests; disconnect/reconnect tests; duplicate-event
checks; generation failures; mobile responsiveness and accessibility.
14. Required Acceptance Tests
Multiplayer: two separate browsers join one room; both see the same state; turns alternate; disconnects do not corrupt the
session.
ASL: real repository pipeline is integrated; webcam access works; a signed letter produces a visible prediction;
confidence/stability checks reduce accidental submissions; confirmation/retry works; accepted letters synchronize; held
gestures do not repeat; limitations are documented.
Gameplay: valid words are detected; timer starts at two minutes; score progresses toward 500; arbitrary client events cannot
manipulate score; points are not duplicated; round ends correctly.
AI imagery: completed word triggers the configured service; both players see the same image; loading and failure states
work; mock generation is clearly labeled.
Communication: chat works in real time; voice transcription can be toggled independently; unsupported features fail
gracefully.
Sharing: accurate statistics appear; the user initiates a supported sharing flow; personal information is not published without
consent.
Quality: responsive and polished UI; no refresh required for the core loop; credentials are not exposed; errors do not crash
the app; README explains setup, dependencies, environment variables, and limitations.

SignSync — AI build prompt and guide
Page 9
15. Final Instructions to the Coding Agent
Before writing code, inspect the existing project and the specified recognition repository. Determine what can be reused and
which external services are accessible. Implement the application in small, verifiable stages.
Prioritize this working demonstration: two players connect → alternate signing real ASL letters → recognized letters appear
on both screens → a word is completed → an AI-generated image appears for both players → the team earns points toward
500 within two minutes.
Once the core experience works, add chat, voice transcription, final statistics, and Facebook sharing. Use modular code,
clear error handling, and a README with setup/testing instructions.
Do not silently substitute mock recognition for the required GitHub model. Do not claim Muse Image or Facebook integration
is operational without testing. If a service is unavailable, explain the limitation and implement the best modular fallback.
The final result should be a working, visually polished hackathon prototype demonstrating how ASL, AI-generated images,
and collaborative play can bring people closer together.

SignSync — AI build prompt and guide
Page 10
Part 3 — Final Demo Checklist
●
Two players can join from separate browsers or devices.

●
The real GitHub-based recognition pipeline predicts letters from webcam input.

●
Players can confirm or retry predictions, and held gestures do not create duplicates.

●
Letters and turns synchronize correctly for both players.

●
A valid word triggers one image-generation request and both players see the same result.

●
The two-minute timer and shared score work, with a clear target of 500 points.

●
Chat works; voice transcription is optional.

●
The results card is accurate and sharing requires explicit user action.

●
Mock services are clearly labeled and never presented as real integrations.

●
The README documents installation, configuration, startup, and known limitations.

Important: The GitHub repository, Muse Image access, and Facebook permissions must be checked in
your actual development environment. This document specifies the intended integration but does not
itself verify credentials or guarantee that external services are available.