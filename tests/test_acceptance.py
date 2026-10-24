import numpy as np

from server.acceptance import HoldTracker, top_letter
from server.classifier import LETTERS


def onehot(letter, conf=1.0):
    p = np.full(26, (1 - conf) / 25, dtype=np.float32)
    p[LETTERS.index(letter)] = conf
    return p


def test_top_letter_masks_j_and_z():
    p = onehot("J", 0.9)
    p[LETTERS.index("A")] = 0.05
    assert top_letter(p)[0] == "A"


def test_accepts_after_a_full_hold():
    h = HoldTracker(0.6, 0.8)
    assert h.update("A", onehot("A"), 0.0) == ("A", 1.0, 0.0, False)
    assert h.update("A", onehot("A"), 0.3)[2] == 0.5
    assert h.update("A", onehot("A"), 0.6)[3] is True


def test_low_confidence_never_accepts():
    h = HoldTracker(0.6, 0.8)
    for i in range(20):
        assert h.update("A", onehot("A", 0.79), i * 0.1)[3] is False


def test_no_hand_resets_the_hold():
    h = HoldTracker(0.6, 0.8)
    h.update("A", onehot("A"), 0.0)
    assert h.update("A", None, 0.3) == (None, 0.0, 0.0, False)
    assert h.update("A", onehot("A"), 0.6)[3] is False  # the hold restarts here
    assert h.update("A", onehot("A"), 1.3)[3] is True


def test_wrong_letter_resets_the_hold():
    h = HoldTracker(0.6, 0.8)
    h.update("A", onehot("A"), 0.0)
    assert h.update("A", onehot("B"), 0.3)[0] == "B"
    assert h.update("A", onehot("A"), 0.6)[3] is False


def test_repeated_letter_needs_a_second_full_hold():
    h = HoldTracker(0.6, 0.8)
    h.update("O", onehot("O"), 0.0)
    assert h.update("O", onehot("O"), 0.6)[3] is True
    assert h.update("O", onehot("O"), 0.7)[3] is False
    assert h.update("O", onehot("O"), 1.3)[3] is True
