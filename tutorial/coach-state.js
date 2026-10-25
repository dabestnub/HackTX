function createCoachState(options = {}) {
  const now = options.now || (() => Date.now());
  const failureThreshold = Number(options.failureThreshold || 2);
  const cooldownMs = Number(options.cooldownMs || 2000);

  const state = {
    enabled: false,
    lessonId: null,
    queue: [],
    index: 0,
    startedAt: null,
    completedAt: null,
    perLetter: new Map(),
    lastAdviceAt: 0,
  };

  function ensureEntry(letter) {
    if (!state.perLetter.has(letter)) {
      state.perLetter.set(letter, {
        attempts: 0,
        failures: 0,
        successes: 0,
        errors: new Map(),
        adviceCount: 0,
      });
    }
    return state.perLetter.get(letter);
  }

  return {
    start(lessonId, letters) {
      state.enabled = true;
      state.lessonId = lessonId || "default";
      state.queue = Array.isArray(letters) ? letters.slice() : [];
      state.index = 0;
      state.startedAt = now();
      state.completedAt = null;
      state.perLetter = new Map();
      state.lastAdviceAt = 0;
      for (const letter of state.queue) ensureEntry(letter);
      return this.snapshot();
    },

    stop() {
      state.enabled = false;
      state.completedAt = now();
      return this.snapshot();
    },

    currentLetter() {
      return state.queue[state.index] || null;
    },

    markAttempt(letter, { success = false, errors = [] } = {}) {
      const entry = ensureEntry(letter);
      entry.attempts += 1;
      if (success) {
        entry.successes += 1;
        if (letter === this.currentLetter()) state.index += 1;
      } else {
        entry.failures += 1;
        for (const code of errors) {
          entry.errors.set(code, (entry.errors.get(code) || 0) + 1);
        }
      }
      if (state.index >= state.queue.length && state.enabled) {
        state.completedAt = now();
      }
      return this.snapshot();
    },

    shouldRequestAdvice(letter) {
      if (!state.enabled) return false;
      if (!letter || letter !== this.currentLetter()) return false;
      const entry = ensureEntry(letter);
      if (entry.failures < failureThreshold) return false;
      if (now() - state.lastAdviceAt < cooldownMs) return false;
      return true;
    },

    markAdviceUsed(letter) {
      const entry = ensureEntry(letter);
      entry.adviceCount += 1;
      state.lastAdviceAt = now();
      return this.snapshot();
    },

    snapshot() {
      return {
        enabled: state.enabled,
        lessonId: state.lessonId,
        currentLetter: this.currentLetter(),
        remaining: Math.max(0, state.queue.length - state.index),
        completed: state.index >= state.queue.length,
        startedAt: state.startedAt,
        completedAt: state.completedAt,
        perLetter: Object.fromEntries(
          [...state.perLetter.entries()].map(([letter, entry]) => [
            letter,
            {
              attempts: entry.attempts,
              failures: entry.failures,
              successes: entry.successes,
              adviceCount: entry.adviceCount,
              errors: Object.fromEntries(entry.errors),
            },
          ])
        ),
      };
    },
  };
}

module.exports = { createCoachState };
