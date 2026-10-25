/* Browser adaptation of JordiEncaboBataller/sign-language-cv's HybridPredictor.
 * Source, pinned revision and model limitations: THIRD_PARTY_NOTICES.md.
 */
(() => {
  function normalizeLandmarks(hand) {
    if (!hand || hand.length !== 21) return null;
    // Upstream casts MediaPipe coordinates to float32 before normalising in float64.
    const raw = Float32Array.from(hand.flatMap(({ x, y, z }) => [x, y, z]));
    if (!raw.every(Number.isFinite)) return null;
    const scale = Math.hypot(raw[27] - raw[0], raw[28] - raw[1], raw[29] - raw[2]);
    if (scale < 1e-6) return null;
    return Float32Array.from(raw, (value, index) => (value - raw[index % 3]) / scale);
  }

  function addVelocityFeatures(frames) {
    const features = new Float32Array(frames.length * 126);
    frames.forEach((frame, index) => {
      features.set(frame, index * 126);
      if (index) {
        for (let feature = 0; feature < 63; feature += 1) {
          features[index * 126 + 63 + feature] = frame[feature] - frames[index - 1][feature];
        }
      }
    });
    return features;
  }

  function validateConfig(config) {
    const matches = (actual, expected) => Array.isArray(actual)
      && actual.length === expected.length && [...actual].sort().join(',') === [...expected].sort().join(',');
    if (!matches(config.static_classes, [...'ABCDEFGHIKLMNOPQRSTUVWXY'])
      || !matches(config.motion_classes, ['J', 'STATIC', 'Z'])
      || config.sequence_length !== 30
      || !Array.isArray(config.scaler_mean) || config.scaler_mean.length !== 63
      || !config.scaler_mean.every(Number.isFinite)
      || !Array.isArray(config.scaler_scale) || config.scaler_scale.length !== 63
      || !config.scaler_scale.every(value => Number.isFinite(value) && value > 0)) {
      throw new Error('ASL model labels or preprocessing settings are invalid.');
    }
  }

  class HybridRecognizer {
    constructor(ort, config, staticSession, motionSession) {
      validateConfig(config);
      this.ort = ort;
      this.config = config;
      this.staticSession = staticSession;
      this.motionSession = motionSession;
      this.generation = 0;
      this.reset();
    }

    static async load(ort) {
      if (!ort) throw new Error('ONNX Runtime Web did not load. Check the connection and reload.');
      const response = await fetch('./assets/asl_model_config.json');
      if (!response.ok) throw new Error(`Could not load ASL model settings (HTTP ${response.status}).`);
      const config = await response.json();
      validateConfig(config);
      const sessions = await Promise.allSettled(['static', 'motion'].map(kind =>
        ort.InferenceSession.create(`./assets/asl_${kind}.onnx`, { executionProviders: ['wasm'] })));
      const failed = sessions.find(result => result.status === 'rejected');
      if (failed) {
        await Promise.all(sessions.filter(result => result.status === 'fulfilled').map(result => result.value.release()));
        throw failed.reason;
      }
      return new HybridRecognizer(ort, config, sessions[0].value, sessions[1].value);
    }

    reset() {
      this.generation += 1;
      this.frames = [];
      this.history = [];
      this.motionStreak = 0;
      this.cooldown = 0;
      this.motionActive = false;
      this.lastTimestamp = null;
      this.pendingMotion = null;
      this.lastConfirmationAt = null;
    }

    confirmPrediction(prediction, hand, timestamp, holdDuration) {
      if (this.lastConfirmationAt !== null
        && (timestamp - this.lastConfirmationAt > 500 || timestamp <= this.lastConfirmationAt)) {
        this.pendingMotion = null;
      }
      this.lastConfirmationAt = timestamp;
      const pose = normalizeLandmarks(hand);
      if (!pose) {
        this.pendingMotion = null;
        return { label: '', score: 0, mode: prediction.mode };
      }
      if (prediction.mode === 'motion' && ['J', 'Z'].includes(prediction.label) && prediction.score >= 0.60) {
        const previous = this.pendingMotion;
        this.pendingMotion = {
          prediction, pose, lastDetectedAt: timestamp,
          firstDetectedAt: previous?.prediction.label === prediction.label ? previous.firstDetectedAt : timestamp,
        };
        return prediction;
      }
      const pending = this.pendingMotion;
      if (pending) {
        const poseDifference = Math.sqrt(pose.reduce((sum, value, index) =>
          sum + (value - pending.pose[index]) ** 2, 0) / pose.length);
        // Retain a detected motion while its ending pose is held, not while another sign is made.
        // shortcut: 0.08 palm lengths of RMS drift is uncalibrated across signers; tune with live-camera samples.
        if (pending.lastDetectedAt - pending.firstDetectedAt >= 200
          && timestamp - pending.lastDetectedAt <= holdDuration
          && poseDifference <= 0.08) {
          return { ...pending.prediction, holdingMotion: true };
        }
        this.pendingMotion = null;
      }
      return prediction;
    }

    async predict(hand, timestamp) {
      if (this.lastTimestamp !== null
        && (timestamp - this.lastTimestamp > 500 || timestamp <= this.lastTimestamp)) this.reset();
      const norm = normalizeLandmarks(hand);
      if (!norm) {
        this.reset();
        return { label: '', score: 0, mode: 'static' };
      }
      this.lastTimestamp = timestamp;
      this.frames.push(norm);
      if (this.frames.length > this.config.sequence_length) this.frames.shift();

      let movement = 0;
      if (this.frames.length >= 10) {
        const recent = this.frames.slice(-10);
        for (let feature = 0; feature < 63; feature += 1) {
          const mean = recent.reduce((sum, frame) => sum + frame[feature], 0) / 10;
          movement += recent.reduce((sum, frame) => sum + (frame[feature] - mean) ** 2, 0) / 10 / 63;
        }
      }
      if (movement >= 0.0005) {
        this.motionStreak += 1;
        this.cooldown = 0;
        if (this.motionStreak >= 8) this.motionActive = true;
      } else {
        this.cooldown += 1;
        this.motionStreak = 0;
        if (this.cooldown >= 20) {
          this.motionActive = false;
          this.cooldown = 0;
        }
      }

      const motion = this.motionActive && this.frames.length === this.config.sequence_length;
      const mode = motion ? 'motion' : 'static';
      const classes = motion ? this.config.motion_classes : this.config.static_classes;
      // Match StandardScaler's two float32 in-place operations.
      const features = motion ? addVelocityFeatures(this.frames) : Float32Array.from(norm,
        (value, index) => Math.fround(value - this.config.scaler_mean[index]) / this.config.scaler_scale[index]);
      const shape = motion ? [1, this.config.sequence_length, 126] : [1, 63];
      const generation = this.generation;
      const session = motion ? this.motionSession : this.staticSession;
      const output = await session.run({ landmarks: new this.ort.Tensor('float32', features, shape) });
      // A pause, new prompt or stopped camera invalidates any in-flight prediction.
      if (generation !== this.generation) return { label: '', score: 0, mode };
      const logits = output.logits.data;
      if (logits.length !== classes.length || !logits.every(Number.isFinite)) {
        throw new Error('ASL classifier returned invalid scores.');
      }
      const max = Math.max(...logits);
      const score = 1 / logits.reduce((sum, value) => sum + Math.exp(value - max), 0);
      let label = classes[logits.indexOf(max)];
      const candidate = label !== 'STATIC' && score < 0.60 ? label : '';
      if (!motion) {
        this.history.push(label);
        if (this.history.length > 10) this.history.shift();
        const counts = new Map();
        for (const item of this.history) counts.set(item, (counts.get(item) || 0) + 1);
        label = [...counts].sort((a, b) => b[1] - a[1])[0][0];
      }
      if (label === 'STATIC' || score < 0.60) label = '';
      return { label, candidate, score, mode };
    }
  }

  const api = { HybridRecognizer, normalizeLandmarks, addVelocityFeatures };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else globalThis.ASLRecognition = api;
})();
