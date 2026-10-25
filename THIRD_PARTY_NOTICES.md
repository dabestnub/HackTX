# ASL recognition attribution

SignSync uses the static MLP and motion LSTM trained by Jordi Encabo Bataller in
[sign-language-cv](https://github.com/JordiEncaboBataller/sign-language-cv), pinned to
commit `6f5e1de849d67788aad8b538c9ef7b5b4401c02c`.

- `assets/asl_static.onnx` is converted from `saved_models/mlp.pt`.
- `assets/asl_motion.onnx` is converted from `saved_models/lstm_jz.pt`.
- `assets/asl_model_config.json` preserves class order, static scaling parameters,
  sequence length, and SHA-256 hashes of the source checkpoints.
- `asl-recognition.js` adapts the normalization in `src/math_utils.py`, velocity
  features in `src/sequence_utils.py`, and `HybridPredictor` in `src/main_app.py`
  to JavaScript and ONNX Runtime Web.
- `scripts/export_asl_model.py` reads the upstream model class definitions and
  converts their trained weights without retraining.

The upstream README declares MIT licensing and links to a LICENSE file, but no
LICENSE file is present at this pinned revision. This notice records that
upstream declaration and attribution; it does not supply missing license text.

Adaptation: predictions run locally in the browser and feed SignSync's existing
1.5-second submission logic. Autocorrect, word/sentence generation and text-to-
speech from the upstream desktop UI are not included. Recognition state resets
on missing hands, invalid landmarks, interrupted frames, paused recognition,
changed prompts and stopped cameras. J/Z require a sequence of actual model
predictions; the prompted letter never substitutes for the model's output.

Upstream limitations: J/Z were trained on a single person's movement, while the
static dataset used a small group. Neither source test accuracy nor export
parity proves performance for new signers or cameras.

MediaPipe Tasks Vision and ONNX Runtime Web remain externally loaded runtimes;
the MediaPipe hand-landmarker asset is downloaded from Google's model hosting.
Their respective upstream licenses continue to apply.
