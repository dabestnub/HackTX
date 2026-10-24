import numpy as np

from .classifier import LETTERS, MOTION_LETTERS

_ALLOWED = np.array([c not in MOTION_LETTERS for c in LETTERS])


def top_letter(probs) -> tuple[str, float]:
    """Best letter with J and Z masked out (Q14). Confidence is the raw softmax value."""
    p = np.where(_ALLOWED, probs, 0.0)
    i = int(p.argmax())
    return LETTERS[i], float(p[i])


class HoldTracker:
    """Accepts the expected letter once it is the top guess at >= min_conf for hold_seconds (Q14, Q15)."""

    def __init__(self, hold_seconds: float, min_conf: float):
        self.hold_seconds = hold_seconds
        self.min_conf = min_conf
        self.since: float | None = None

    def reset(self):
        self.since = None

    def update(self, expected: str, probs, now: float) -> tuple[str | None, float, float, bool]:
        """Returns (guess, confidence, hold progress 0-1, accepted). probs is None when no hand is seen."""
        if probs is None:
            self.since = None
            return None, 0.0, 0.0, False
        guess, conf = top_letter(probs)
        if guess != expected or conf < self.min_conf:
            self.since = None
            return guess, conf, 0.0, False
        if self.since is None:
            self.since = now
        progress = min(1.0, (now - self.since) / self.hold_seconds)
        if progress < 1.0:
            return guess, conf, progress, False
        self.since = None  # a repeated letter needs a fresh full hold (Q15)
        return guess, conf, 1.0, True
