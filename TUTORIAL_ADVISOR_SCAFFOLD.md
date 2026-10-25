# Tutorial / Advisor scaffold (new files only)

This scaffold adds standalone modules for a future AI tutorial coach without touching existing game files.

## Frontend modules

- `tutorial/coach-state.js`: lesson state machine and advice request cadence.
- `tutorial/landmark-rules.js`: deterministic handshape checks and hint keys.
- `tutorial/advice-client.js`: cached/throttled call wrapper for `/api/coach`.
- `tutorial/tutorial-ui.js`: injectable floating tutorial panel.

## Backend modules

- `server/coach/coach-prompt.js`: Muse text prompt builder and output sanitizer.
- `server/coach/coach-fallback.js`: deterministic fallback hints.
- `server/coach/coach-route.js`: standalone HTTP request handler.

## Tests

- `tutorial/landmark-rules.test.js`
- `tutorial/advice-client.test.js`
- `server/coach/coach-route.test.js`

## Integration (later)

After your team merges settle, do a tiny wiring commit:

1. In `server.js`: route `POST /api/coach` to `createCoachHandler(...)`.
2. In your browser bootstrap: initialize `createCoachState`, `createAdviceClient`, and `createTutorialOverlay`.

No current gameplay files were modified for this scaffold step.
