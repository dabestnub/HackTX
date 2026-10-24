import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # order of the repo's classes.pkl
MOTION_LETTERS = ("J", "Z")  # signed with motion, so a single frame can't show them (Q6)


def features(points) -> np.ndarray:
    """21 (x, y, z) landmarks -> the repo's 63-value input (app.py:56-64)."""
    x = np.asarray(points, dtype=np.float32).reshape(63)
    x = x - x.min()  # one global min over all 63 values, as the repo does
    if x.max() != 0:
        x = x / x.max()
    return x


class Classifier:
    """The repo's GestureModel in numpy, with BatchNorm folded into the Linear layers."""

    def __init__(self, path):
        data = np.load(path)
        self.layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(4)]

    def probs(self, x: np.ndarray) -> np.ndarray:
        """Softmax over the 26 letters, for one (63,) vector or an (N, 63) batch."""
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
