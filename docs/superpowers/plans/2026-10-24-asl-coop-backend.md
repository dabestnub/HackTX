# ASL Co-op Game Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI + WebSocket server on fly.io that runs two-player ASL fingerspelling rounds: it recognizes letters from camera frames, runs rooms, timers and scoring, and gets word pairs and images from Muse.

**Architecture:** A pure game engine (`server/game.py`) takes events with an explicit `now` and returns messages and effects. A thin I/O layer (`server/app.py`) owns sockets, per-player recognizer threads, a 10 Hz room ticker, and Muse calls. Recognition reproduces the repo's `app.py` exactly (flip, MediaPipe Tasks HandLandmarker, 63 min-max features), with the classifier exported once from torch to numpy.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, mediapipe 1.1.x (brings opencv-contrib-python), numpy, httpx, pytest. Torch is used only by the one-time export script.

**Spec:** `handoff.md` in the project root. Sections 4 and 12 are confirmed decisions; do not change them.

**Rules for every task:**
- Run every command from the project root `/DATA2/linke/projects/HackTx`.
- Use `uv run ...` for all Python. Never `pip install` into the system Python.
- Commit locally after each task. Never push. Never commit `Research ASL Models.html`, `.env`, or any API key.

---

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `.python-version`, `uv.lock` | uv project, Python 3.12 |
| `server/config.py` | `Settings` (all tunables) and `MODELS` path |
| `server/classifier.py` | `features()` (repo preprocessing) and the numpy `Classifier` |
| `server/acceptance.py` | `top_letter()` (J/Z masked) and `HoldTracker` (0.8 conf, 0.6 s hold) |
| `server/words.py` | Word regex, fallback adjective and noun lists |
| `server/recognizer.py` | JPEG -> flip -> HandLandmarker -> probabilities and overlay points |
| `server/game.py` | Pure room engine: phases, rounds, timers, scoring, reconnect |
| `server/muse.py` | Muse word-pair and image calls, placeholder SVG |
| `server/app.py` | `create_app()`: WebSocket protocol, room registry, images, /dev, /healthz |
| `server/main.py` | `app = create_app()` for uvicorn |
| `server/static/dev.html` | Minimal camera test page |
| `scripts/export_weights.py` | One-time torch -> numpy export, writes the parity fixture |
| `models/hand_landmarker.task`, `models/classifier.npz` | Model files (committed) |
| `tests/...` | pytest suites; `tests/fixtures/parity.npz` |
| `docs/protocol.md` | WebSocket contract for the frontend teammate |
| `Dockerfile`, `.dockerignore`, `fly.toml` | Deployment |

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `.python-version`, `server/__init__.py`, `server/config.py`
- Create (copied): `models/hand_landmarker.task`

- [x] **Step 1: Initialize git and the ignore file**

```bash
git init -b backend-shit
git remote add origin git@github.com:dabestnub/HackTX.git
```

The controller pushes this branch later; implementers never push.

Write `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.cache/
.env
Research ASL Models.html
```

- [x] **Step 2: Write `pyproject.toml` and `.python-version`**

`pyproject.toml`:

```toml
[project]
name = "asl-coop-backend"
version = "0.1.0"
description = "Backend for a two-player ASL fingerspelling game"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "mediapipe>=1.1",
    "numpy>=1.26",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.python-version`:

```
3.12
```

- [x] **Step 3: Install and check imports**

Run: `uv sync && uv run python -c "import fastapi, uvicorn, httpx, mediapipe, numpy, cv2; print('ok', mediapipe.__version__)"`
Expected: last line `ok 1.1.1` (or a newer 1.x). Do not add `opencv-python` or `opencv-python-headless`; mediapipe already brings `opencv-contrib-python`, and two OpenCV wheels conflict.

- [x] **Step 4: Fetch the repo's model files**

```bash
git clone -q https://github.com/Gj-345/asl-hand-gesture-recognition.git .cache/asl-repo
git -C .cache/asl-repo checkout -q 9de5b22
mkdir -p models
cp .cache/asl-repo/ml/hand_landmarker.task models/hand_landmarker.task
ls -l models/hand_landmarker.task .cache/asl-repo/ml/models/best_model.pth
```

Expected: `hand_landmarker.task` is about 7.8 MB and `best_model.pth` about 326 KB.

- [x] **Step 5: Write `server/__init__.py` (empty) and `server/config.py`**

`server/config.py`:

```python
import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Settings:
    """Game and server tunables. Tests pass faster values; production uses the defaults."""

    time_limit: float = 10.0       # seconds per round for the whole word (Q13)
    hold_seconds: float = 0.6      # Q14
    min_confidence: float = 0.8    # Q14
    countdown: float = 3.0
    reveal: float = 5.0            # seconds an image shows before the next countdown
    reconnect_grace: float = 30.0  # Q8
    points_per_round: int = 100    # Q3
    target_score: int = 500        # Q3
    image_timeout: float = 10.0    # Q19
    word_timeout: float = 8.0
    max_frame_bytes: int = 300_000
    max_frame_width: int = 640     # Q16
    max_frame_height: int = 480
    api_key: str | None = field(default_factory=lambda: os.environ.get("MODEL_API_KEY") or None)
    muse_base_url: str = "https://api.meta.ai/v1"
```

Run: `uv run python -c "from server.config import Settings, MODELS; print(Settings(api_key=None).time_limit, MODELS.name)"`
Expected: `10.0 models`

- [x] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml .python-version uv.lock server/__init__.py server/config.py models/hand_landmarker.task handoff.md docs/superpowers/plans/2026-10-24-asl-coop-backend.md
git commit -m "chore: scaffold uv project and add hand landmarker model"
```

---

### Task 2: Numpy classifier, weight export, parity test

**Files:**
- Create: `scripts/export_weights.py`, `server/classifier.py`, `tests/test_classifier.py`
- Create (generated): `models/classifier.npz`, `tests/fixtures/parity.npz`

- [x] **Step 1: Write the export script**

`scripts/export_weights.py`:

```python
"""One-time export of the repo's torch classifier to numpy (Q18).

    uv run --no-project -p 3.12 --with torch --with numpy \
        --index https://download.pytorch.org/whl/cpu \
        python scripts/export_weights.py .cache/asl-repo/ml/models/best_model.pth

Writes models/classifier.npz (BatchNorm folded into the Linear layers)
and tests/fixtures/parity.npz (inputs plus torch's softmax outputs).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


def build_model():
    # Same layers as GestureModel.network in the repo's ml/model.py
    return nn.Sequential(
        nn.Linear(63, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 26),
    )


def main(pth):
    state = torch.load(pth, map_location="cpu")  # saved on cuda:0
    model = build_model()
    model.load_state_dict({k.removeprefix("network."): v for k, v in state.items()})
    model.eval()

    out = {}
    for i, (lin, bn) in enumerate([(0, 1), (4, 5), (8, 9), (12, None)]):
        w = model[lin].weight.detach().numpy().astype(np.float64)
        b = model[lin].bias.detach().numpy().astype(np.float64)
        if bn is not None:
            m = model[bn]
            scale = m.weight.detach().numpy() / np.sqrt(m.running_var.numpy() + m.eps)
            w = w * scale[:, None]
            b = (b - m.running_mean.numpy()) * scale + m.bias.detach().numpy()
        out[f"w{i}"] = w.T.astype(np.float32)  # stored (in, out) so inference is x @ w + b
        out[f"b{i}"] = b.astype(np.float32)
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez(ROOT / "models" / "classifier.npz", **out)

    rng = np.random.default_rng(0)
    inputs = rng.random((32, 63), dtype=np.float32)
    inputs = inputs - inputs.min(axis=1, keepdims=True)
    inputs = inputs / inputs.max(axis=1, keepdims=True)  # same 0-1 range as real features
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(inputs)), dim=1).numpy()
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "tests" / "fixtures" / "parity.npz", inputs=inputs, probs=probs)
    print("wrote models/classifier.npz and tests/fixtures/parity.npz")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [x] **Step 2: Run the export**

Run:

```bash
uv run --no-project -p 3.12 --with torch --with numpy --index https://download.pytorch.org/whl/cpu python scripts/export_weights.py .cache/asl-repo/ml/models/best_model.pth
```

Expected: `wrote models/classifier.npz and tests/fixtures/parity.npz` (the first run downloads CPU torch, about 200 MB).

- [x] **Step 3: Write the failing tests**

`tests/test_classifier.py`:

```python
from pathlib import Path

import numpy as np

from server.classifier import Classifier, features
from server.config import MODELS

FIXTURES = Path(__file__).parent / "fixtures"


def test_features_scale_to_unit_range():
    points = np.arange(63, dtype=np.float32).reshape(21, 3) * 0.01 - 0.2
    x = features(points)
    assert x.shape == (63,)
    assert x.min() == 0.0 and x.max() == 1.0


def test_features_of_identical_points_stay_zero():
    assert not features(np.full((21, 3), 0.5)).any()


def test_numpy_classifier_matches_torch():
    clf = Classifier(MODELS / "classifier.npz")
    fx = np.load(FIXTURES / "parity.npz")
    probs = clf.probs(fx["inputs"])
    np.testing.assert_allclose(probs, fx["probs"], atol=1e-5)
    assert (probs.argmax(axis=1) == fx["probs"].argmax(axis=1)).all()


def test_single_vector_gives_26_probabilities():
    p = Classifier(MODELS / "classifier.npz").probs(np.zeros(63, dtype=np.float32))
    assert p.shape == (26,)
    assert abs(float(p.sum()) - 1.0) < 1e-5
```

- [x] **Step 4: Run the tests to see them fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'server.classifier'`

- [x] **Step 5: Implement `server/classifier.py`**

```python
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
```

- [x] **Step 6: Run the tests to see them pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: 4 passed

- [x] **Step 7: Commit**

```bash
git add scripts/export_weights.py server/classifier.py tests/test_classifier.py models/classifier.npz tests/fixtures/parity.npz
git commit -m "feat: numpy classifier exported from the repo's torch model, with parity test"
```

---

### Task 3: Letter acceptance rule

**Files:**
- Create: `server/acceptance.py`, `tests/test_acceptance.py`

- [x] **Step 1: Write the failing tests**

`tests/test_acceptance.py`:

```python
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
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_acceptance.py -v`
Expected: collection error, `No module named 'server.acceptance'`

- [x] **Step 3: Implement `server/acceptance.py`**

```python
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
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_acceptance.py -v`
Expected: 6 passed

- [x] **Step 5: Commit**

```bash
git add server/acceptance.py tests/test_acceptance.py
git commit -m "feat: letter acceptance rule (0.8 confidence held 0.6 s, J/Z masked)"
```

---

### Task 4: Word validation and fallback words

**Files:**
- Create: `server/words.py`, `tests/test_words.py`

- [x] **Step 1: Write the failing tests**

`tests/test_words.py`:

```python
import random

import pytest

from server import words


def test_fallback_words_are_all_valid_and_disjoint():
    for w in words.ADJECTIVES + words.NOUNS:
        assert words.is_valid_word(w, set()), w
    assert not set(words.ADJECTIVES) & set(words.NOUNS)
    assert len(words.ADJECTIVES) >= 25 and len(words.NOUNS) >= 25


@pytest.mark.parametrize("word", ["JAM", "ZOO", "AT", "PURPLE", "red", "RE D", "CAT1", ""])
def test_invalid_words_are_rejected(word):
    assert not words.is_valid_word(word, set())


def test_used_words_are_rejected():
    assert not words.is_valid_word("CAT", {"CAT"})


def test_pair_needs_two_different_valid_words():
    assert words.valid_pair("RED", "CAT", set())
    assert not words.valid_pair("CAT", "CAT", set())
    assert not words.valid_pair("RED", "ZEBRA", set())


def test_normalize_trims_and_uppercases():
    assert words.normalize("  red ") == "RED"


def test_fallback_pair_avoids_used_words():
    used = set(words.ADJECTIVES[1:]) | set(words.NOUNS[1:])
    assert words.fallback_pair(used, random.Random(0)) == (words.ADJECTIVES[0], words.NOUNS[0])
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_words.py -v`
Expected: collection error, `cannot import name 'words'`

- [x] **Step 3: Implement `server/words.py`**

```python
import random
import re

WORD_RE = re.compile(r"[A-IK-Y]{3,5}")  # A-Z without J and Z (Q6), 3 to 5 letters (Q7)

ADJECTIVES = [
    "RED", "BLUE", "PINK", "GOLD", "GREEN", "BLACK", "WHITE", "BIG", "TINY", "HAPPY",
    "SAD", "FAST", "SLOW", "WET", "HOT", "COLD", "SILLY", "FUNNY", "ANGRY", "SHY",
    "OLD", "CUTE", "LOUD", "SOFT", "SHINY", "BRAVE", "SPICY", "ROYAL", "TALL",
]
NOUNS = [
    "CAT", "DOG", "FISH", "BIRD", "FROG", "BEAR", "LION", "TIGER", "HORSE", "MOUSE",
    "DUCK", "COW", "PIG", "OWL", "FOX", "GOAT", "SHEEP", "CAKE", "APPLE", "TACO",
    "ROBOT", "HAT", "BOAT", "CAR", "TRAIN", "HOUSE", "TREE", "MOON", "STAR", "BOOK",
    "SHOE", "SOCK", "CHAIR", "CLOUD", "GHOST", "CROWN", "PLANT", "BUS", "KITE", "DRUM",
]


def normalize(word: str) -> str:
    return str(word).strip().upper()


def is_valid_word(word: str, used: set[str]) -> bool:
    return WORD_RE.fullmatch(word) is not None and word not in used


def valid_pair(adjective: str, noun: str, used: set[str]) -> bool:
    return adjective != noun and is_valid_word(adjective, used) and is_valid_word(noun, used)


def fallback_pair(used: set[str], rng=random) -> tuple[str, str]:
    """A built-in pair for when Muse is off, slow, or returns bad words (Q7)."""
    # shortcut: repeats words once a game uses all of a list (29+ rounds); fine for 5-round games
    adjectives = [w for w in ADJECTIVES if w not in used] or ADJECTIVES
    nouns = [w for w in NOUNS if w not in used] or NOUNS
    return rng.choice(adjectives), rng.choice(nouns)
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_words.py -v`
Expected: 13 passed

- [x] **Step 5: Commit**

```bash
git add server/words.py tests/test_words.py
git commit -m "feat: word validation and fallback word lists"
```

---

### Task 5: Recognizer (MediaPipe)

**Files:**
- Create: `server/recognizer.py`, `tests/test_recognizer.py`

- [x] **Step 1: Write the failing tests**

`tests/test_recognizer.py`:

```python
import cv2
import numpy as np
import pytest

from server.classifier import Classifier
from server.config import MODELS
from server.recognizer import Recognizer


@pytest.fixture(scope="module")
def rec():
    r = Recognizer(Classifier(MODELS / "classifier.npz"))
    yield r
    r.close()


def jpeg(width, height):
    return cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))[1].tobytes()


def test_blank_frame_has_no_hand(rec):
    r = rec.process(jpeg(640, 480))
    assert r.probs is None and r.landmarks is None


def test_garbage_bytes_are_rejected(rec):
    assert rec.process(b"not a jpeg") is None
    assert rec.process(b"") is None


def test_oversized_frame_is_rejected(rec):
    assert rec.process(jpeg(1280, 720)) is None
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_recognizer.py -v`
Expected: collection error, `No module named 'server.recognizer'`

- [x] **Step 3: Implement `server/recognizer.py`**

```python
"""Camera frame -> MediaPipe hand landmarks -> letter probabilities, matching the repo's app.py."""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .classifier import Classifier, features
from .config import MODELS


@dataclass
class Recognition:
    probs: np.ndarray | None  # 26 softmax values, or None when no hand is seen
    landmarks: list | None    # 21 [x, y] points, 0-1, in the mirrored (selfie) view


class Recognizer:
    """One per player; the HandLandmarker is never shared between players."""

    def __init__(self, classifier: Classifier, task_path=MODELS / "hand_landmarker.task", max_w=640, max_h=480):
        options = vision.HandLandmarkerOptions(  # same settings as app.py:30-37
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.classifier = classifier
        self.max_w, self.max_h = max_w, max_h

    def process(self, jpeg: bytes) -> Recognition | None:
        """None when the bytes are not a JPEG within the size limit (Q16)."""
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        if frame is None or frame.shape[1] > self.max_w or frame.shape[0] > self.max_h:
            return None
        frame = cv2.flip(frame, 1)  # app.py:105 mirrors every frame before MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return Recognition(None, None)
        hand = result.hand_landmarks[0]  # first hand only, handedness ignored, as the repo does
        x = features([[lm.x, lm.y, lm.z] for lm in hand])
        return Recognition(self.classifier.probs(x), [[round(lm.x, 3), round(lm.y, 3)] for lm in hand])

    def close(self):
        self.detector.close()
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_recognizer.py -v`
Expected: 3 passed (MediaPipe log lines on stderr are normal).

- [x] **Step 5: Commit**

```bash
git add server/recognizer.py tests/test_recognizer.py
git commit -m "feat: MediaPipe recognizer matching the repo's live app"
```

---

### Task 6: Game engine 1/3, rooms, lobby, countdown, round start

**Files:**
- Create: `server/game.py`, `tests/test_game.py`

- [x] **Step 1: Write the failing tests**

`tests/test_game.py`:

```python
import random

import numpy as np

from server import words
from server.classifier import LETTERS
from server.config import Settings
from server.game import CloseRoom, FetchPair, GenerateImage, Phase, Room, Send

S = Settings(api_key=None)


def sent(outs, kind, to="any"):
    return [o.msg for o in outs if isinstance(o, Send) and o.msg["type"] == kind and (to == "any" or o.to == to)]


def onehot(letter):
    p = np.zeros(26, dtype=np.float32)
    p[LETTERS.index(letter)] = 1.0
    return p


def lobby():
    room = Room("ABCD", S, random.Random(0))
    room.join("Ann", "t1", 0.0)
    room.join("Ben", "t2", 0.0)
    return room


def start(room, pair=("RED", "CAT"), now=0.0):
    """Ready both players, supply the pair, finish the countdown. Returns the round-start outputs."""
    room.set_ready(1, True, now)
    room.set_ready(2, True, now)
    room.on_pair(*pair)
    return room.tick(now + S.countdown)


def sign(room, slot, word, t):
    """Holds each letter a little past a full hold. Returns (outputs, time after the word)."""
    outs = []
    for letter in word:
        outs += room.on_recognition(slot, onehot(letter), None, t)
        t += S.hold_seconds + 0.05
        outs += room.on_recognition(slot, onehot(letter), None, t)
        t += 0.1
    return outs, t


def test_join_assigns_slots_and_tokens():
    room = Room("ABCD", S)
    slot, outs = room.join("Ann", "t1", 0.0)
    assert slot == 1
    assert sent(outs, "room_joined", to=1) == [{"type": "room_joined", "room": "ABCD", "slot": 1, "token": "t1"}]
    assert sent(outs, "room_state")[0]["players"][0]["name"] == "Ann"
    assert room.join("Ben", "t2", 0.0)[0] == 2
    assert room.full
    assert room.slot_for_token("t2") == 2 and room.slot_for_token("nope") is None


def test_game_starts_when_both_ready():
    room = lobby()
    outs = room.set_ready(1, True, 0.0)
    assert room.phase is Phase.LOBBY and sent(outs, "room_state")
    outs = room.set_ready(2, True, 0.0)
    assert room.phase is Phase.COUNTDOWN
    assert sent(outs, "countdown") == [{"type": "countdown", "round": 1, "seconds": 3.0}]
    assert any(isinstance(o, FetchPair) for o in outs)


def test_countdown_waits_for_its_deadline():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    assert room.tick(2.9) == []
    assert room.phase is Phase.COUNTDOWN


def test_round_start_gives_each_player_a_word():
    room = lobby()
    outs = start(room, ("RED", "CAT"))
    assert room.phase is Phase.SIGNING
    first = sent(outs, "round_start", to=1)[0]
    assert first["your_word"] == "RED"  # slot 1 signs the adjective in round 1
    assert first["phrase"] == "RED CAT"
    assert first["words"] == [{"slot": 1, "word": "RED"}, {"slot": 2, "word": "CAT"}]
    assert sent(outs, "round_start", to=2)[0]["your_word"] == "CAT"
    assert sent(outs, "room_state")[0]["time_left"] == 10.0
    assert [o.used for o in outs if isinstance(o, FetchPair)] == [{"RED", "CAT"}]


def test_invalid_pair_from_muse_falls_back_to_builtin_words():
    room = lobby()
    outs = start(room, ("JAZZY", "CAT"))  # J and Z are never allowed (Q6)
    assert sent(outs, "round_start", to=1)[0]["your_word"] in words.ADJECTIVES


def test_only_the_first_valid_pair_is_kept():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    room.on_pair("red", "cat")
    room.on_pair("BIG", "DOG")
    outs = room.tick(S.countdown)
    assert sent(outs, "round_start", to=1)[0]["phrase"] == "RED CAT"


def test_ready_is_ignored_outside_the_lobby():
    room = lobby()
    start(room)
    assert room.set_ready(1, False, 4.0) == []
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_game.py -v`
Expected: collection error, `No module named 'server.game'`

- [x] **Step 3: Implement `server/game.py`**

```python
"""Pure game engine for one room: events in, messages and effects out.

No I/O and no clock. Every time-dependent method takes `now` (time.monotonic() seconds).
server/app.py delivers the Send messages and performs the FetchPair, GenerateImage and CloseRoom effects.
"""
import random
from dataclasses import dataclass
from enum import Enum

from . import words
from .acceptance import HoldTracker
from .config import Settings


class Phase(str, Enum):
    LOBBY = "lobby"
    COUNTDOWN = "countdown"
    SIGNING = "signing"
    GENERATING = "generating"
    REVEAL = "reveal"
    GAME_OVER = "game_over"


@dataclass
class Send:
    to: int | None  # slot 1 or 2; None means both players
    msg: dict


@dataclass
class FetchPair:
    used: frozenset


@dataclass
class GenerateImage:
    round_no: int
    phrase: str


@dataclass
class CloseRoom:
    reason: str


@dataclass
class Player:
    slot: int
    name: str
    token: str
    hold: HoldTracker
    connected: bool = True
    ready: bool = False
    word: str = ""
    index: int = 0

    @property
    def done(self) -> bool:
        return bool(self.word) and self.index >= len(self.word)


class Room:
    def __init__(self, code: str, settings: Settings, rng=random):
        self.code = code
        self.s = settings
        self.rng = rng
        self.players: dict[int, Player] = {}
        self.phase = Phase.LOBBY
        self.score = 0
        self.round_no = 0
        self.phrase = ""
        self.deadline: float | None = None   # when the current timed phase ends
        self.paused_at: float | None = None  # set while a player is disconnected (Q8)
        self.close_at: float | None = None
        self.used: set[str] = set()
        self.next_pair: tuple[str, str] | None = None
        self.gallery: list[dict] = []
        self.closed = False

    # ---- membership ----

    @property
    def full(self) -> bool:
        return len(self.players) == 2

    def join(self, name: str, token: str, now: float) -> tuple[int, list]:
        slot = 1 if 1 not in self.players else 2
        self.players[slot] = Player(slot, name, token, HoldTracker(self.s.hold_seconds, self.s.min_confidence))
        return slot, [Send(slot, self._joined(slot)), self._state(now)]

    def slot_for_token(self, token: str) -> int | None:
        return next((p.slot for p in self.players.values() if p.token == token), None)

    # ---- lobby and words ----

    def set_ready(self, slot: int, ready: bool, now: float) -> list:
        if self.phase is not Phase.LOBBY:
            return []
        self.players[slot].ready = ready
        if self.full and all(p.ready and p.connected for p in self.players.values()):
            return [FetchPair(frozenset(self.used)), *self._start_countdown(now)]
        return [self._state(now)]

    def on_pair(self, adjective: str, noun: str) -> list:
        adjective, noun = words.normalize(adjective), words.normalize(noun)
        if self.next_pair is None and words.valid_pair(adjective, noun, self.used):
            self.next_pair = (adjective, noun)
        return []

    # ---- clock ----

    def tick(self, now: float) -> list:
        if self.closed or self.deadline is None or now < self.deadline:
            return []
        if self.phase is Phase.COUNTDOWN:
            return self._start_signing(now)
        return []

    # ---- transitions ----

    def _start_countdown(self, now: float) -> list:
        self.phase = Phase.COUNTDOWN
        self.deadline = now + self.s.countdown
        for p in self.players.values():
            p.word, p.index = "", 0
            p.hold.reset()
        return [Send(None, {"type": "countdown", "round": self.round_no + 1, "seconds": self.s.countdown}),
                self._state(now)]

    def _start_signing(self, now: float) -> list:
        self.round_no += 1
        adjective, noun = self.next_pair or words.fallback_pair(self.used, self.rng)
        self.next_pair = None
        self.used |= {adjective, noun}
        self.phrase = f"{adjective} {noun}"
        # Slot 1 signs the adjective in odd rounds; players swap each round (Q7)
        first, second = (adjective, noun) if self.round_no % 2 else (noun, adjective)
        self.players[1].word, self.players[2].word = first, second
        self.phase = Phase.SIGNING
        self.deadline = now + self.s.time_limit
        starts = [Send(p.slot, self._round_start(p.slot)) for p in self._ordered()]
        return [*starts, self._state(now), FetchPair(frozenset(self.used))]  # prefetch the next pair

    # ---- messages ----

    def _ordered(self) -> list[Player]:
        return [self.players[s] for s in sorted(self.players)]

    def _joined(self, slot: int) -> dict:
        return {"type": "room_joined", "room": self.code, "slot": slot, "token": self.players[slot].token}

    def _round_start(self, slot: int) -> dict:
        return {"type": "round_start", "round": self.round_no, "phrase": self.phrase,
                "words": [{"slot": p.slot, "word": p.word} for p in self._ordered()],
                "your_word": self.players[slot].word, "time_limit": self.s.time_limit}

    def _time_left(self, now: float) -> float | None:
        if self.phase is not Phase.SIGNING or self.deadline is None:
            return None
        ref = self.paused_at if self.paused_at is not None else now
        return round(max(0.0, self.deadline - ref), 2)

    def _state(self, now: float) -> Send:
        return Send(None, {
            "type": "room_state", "room": self.code, "phase": self.phase.value,
            "paused": self.paused_at is not None, "score": self.score, "target": self.s.target_score,
            "round": self.round_no, "time_left": self._time_left(now),
            "players": [{"slot": p.slot, "name": p.name, "ready": p.ready, "connected": p.connected}
                        for p in self._ordered()],
        })
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_game.py -v`
Expected: 7 passed

- [x] **Step 5: Commit**

```bash
git add server/game.py tests/test_game.py
git commit -m "feat: game engine rooms, lobby, countdown and round start"
```

---

### Task 7: Game engine 2/3, signing, images, scoring, game over

**Files:**
- Modify: `server/game.py` (add methods, replace `tick`)
- Modify: `tests/test_game.py` (append tests)

- [x] **Step 1: Append the failing tests to `tests/test_game.py`**

```python
def test_recognition_feedback_goes_to_the_signer_only():
    room = lobby()
    start(room)
    outs = room.on_recognition(1, onehot("R"), [[0.5, 0.5]] * 21, 3.0)
    assert [o.to for o in outs] == [1]
    [msg] = sent(outs, "recognition")
    assert msg == {"type": "recognition", "guess": "R", "confidence": 1.0, "hold": 0.0,
                   "accepted": False, "expected": "R", "landmarks": [[0.5, 0.5]] * 21}


def test_wrong_letters_are_ignored_without_penalty():
    room = lobby()
    start(room)
    for i in range(10):
        room.on_recognition(1, onehot("B"), None, 3.0 + i * 0.1)
    assert room.players[1].index == 0 and room.score == 0


def test_signing_both_words_requests_the_image():
    room = lobby()
    start(room)
    outs1, _ = sign(room, 1, "RED", 3.0)
    assert [m["done"] for m in sent(outs1, "progress")] == [1, 2, 3]
    assert all(m["slot"] == 1 and m["length"] == 3 for m in sent(outs1, "progress"))
    assert room.phase is Phase.SIGNING
    outs2, _ = sign(room, 2, "CAT", 3.0)
    assert room.phase is Phase.GENERATING
    assert [(o.round_no, o.phrase) for o in outs2 if isinstance(o, GenerateImage)] == [(1, "RED CAT")]
    assert sent(outs2, "image_pending") == [{"type": "image_pending", "round": 1, "phrase": "RED CAT"}]


def test_repeated_letters_need_two_holds():
    room = lobby()
    start(room, ("BIG", "BOOK"))
    outs, _ = sign(room, 2, "BOOK", 3.0)
    assert [m["done"] for m in sent(outs, "progress")] == [1, 2, 3, 4]


def test_frames_outside_signing_are_ignored():
    room = lobby()
    assert room.on_recognition(1, onehot("R"), None, 0.0) == []


def test_image_awards_points_then_reveal_then_next_round_swaps_roles():
    room = lobby()
    start(room)
    sign(room, 1, "RED", 3.0)
    _, t = sign(room, 2, "CAT", 3.0)
    outs = room.on_image(1, "/images/ABCD/1", t)
    assert sent(outs, "image_ready") == [{"type": "image_ready", "round": 1, "phrase": "RED CAT", "url": "/images/ABCD/1"}]
    assert sent(outs, "round_complete") == [{"type": "round_complete", "round": 1, "phrase": "RED CAT",
                                             "points": 100, "score": 100}]
    assert room.phase is Phase.REVEAL
    room.on_pair("BIG", "DOG")
    outs = room.tick(t + S.reveal)
    assert room.phase is Phase.COUNTDOWN and sent(outs, "countdown")[0]["round"] == 2
    outs = room.tick(t + S.reveal + S.countdown)
    assert sent(outs, "round_start", to=1)[0]["your_word"] == "DOG"  # slot 1 signs the noun in round 2


def test_stale_image_is_ignored():
    room = lobby()
    start(room)
    assert room.on_image(99, "/images/ABCD/99", 4.0) == []


def test_five_rounds_reach_500_then_game_over_and_play_again():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    t = 0.0
    pairs = [("RED", "CAT"), ("BIG", "DOG"), ("SAD", "FROG"), ("HOT", "CAKE"), ("OLD", "BOAT")]
    for i, pair in enumerate(pairs, 1):
        room.on_pair(*pair)
        t += S.countdown
        room.tick(t)
        _, t1 = sign(room, 1, room.players[1].word, t)
        _, t2 = sign(room, 2, room.players[2].word, t)
        t = max(t1, t2)
        outs = room.on_image(i, f"/images/ABCD/{i}", t)
        if i < len(pairs):
            t += S.reveal
            room.tick(t)
    assert room.phase is Phase.GAME_OVER
    [over] = sent(outs, "game_over")
    assert over["score"] == 500 and len(over["gallery"]) == 5
    assert over["gallery"][0] == {"phrase": "RED CAT", "url": "/images/ABCD/1"}
    room.play_again(1, t)
    assert room.phase is Phase.LOBBY and room.score == 0 and room.round_no == 0 and not room.used
    assert not any(p.ready for p in room.players.values())
```

- [x] **Step 2: Run the tests to see the new ones fail**

Run: `uv run pytest tests/test_game.py -v`
Expected: the 8 new tests fail with `AttributeError: 'Room' object has no attribute 'on_recognition'` (or `on_image`); the 7 old tests pass.

- [x] **Step 3: Add the methods to `Room` in `server/game.py`**

Insert this block after `on_pair` (before the `# ---- clock ----` section):

```python
    # ---- signing and images ----

    def on_recognition(self, slot: int, probs, landmarks, now: float) -> list:
        p = self.players.get(slot)
        if self.phase is not Phase.SIGNING or self.paused_at is not None or p is None or p.done:
            return []
        expected = p.word[p.index]
        guess, conf, hold, accepted = p.hold.update(expected, probs, now)
        if accepted:
            p.index += 1
        outs = [Send(slot, {"type": "recognition", "guess": guess, "confidence": round(conf, 3),
                            "hold": round(hold, 2), "accepted": accepted, "expected": expected,
                            "landmarks": landmarks})]
        if not accepted:
            return outs
        outs.append(Send(None, {"type": "progress", "slot": slot, "done": p.index, "length": len(p.word)}))
        if all(q.done for q in self.players.values()):
            self.phase = Phase.GENERATING
            self.deadline = None
            outs += [Send(None, {"type": "image_pending", "round": self.round_no, "phrase": self.phrase}),
                     self._state(now), GenerateImage(self.round_no, self.phrase)]
        return outs

    def on_image(self, round_no: int, url: str, now: float) -> list:
        if self.phase is not Phase.GENERATING or round_no != self.round_no:
            return []
        self.score += self.s.points_per_round
        self.gallery.append({"phrase": self.phrase, "url": url})
        outs = [Send(None, {"type": "image_ready", "round": round_no, "phrase": self.phrase, "url": url}),
                Send(None, {"type": "round_complete", "round": round_no, "phrase": self.phrase,
                            "points": self.s.points_per_round, "score": self.score})]
        if self.score >= self.s.target_score:
            self.phase = Phase.GAME_OVER
            self.deadline = None
            outs.append(Send(None, self._game_over()))
        else:
            self.phase = Phase.REVEAL
            self.deadline = now + self.s.reveal
        return [*outs, self._state(now)]

    def play_again(self, slot: int, now: float) -> list:
        if self.phase is not Phase.GAME_OVER:
            return []
        self.phase = Phase.LOBBY
        self.score, self.round_no, self.phrase = 0, 0, ""
        self.used, self.gallery, self.next_pair = set(), [], None
        for p in self.players.values():
            p.ready, p.word, p.index = False, "", 0
        return [self._state(now)]
```

Replace the whole `tick` method with:

```python
    def tick(self, now: float) -> list:
        if self.closed or self.deadline is None or now < self.deadline:
            return []
        if self.phase is Phase.COUNTDOWN:
            return self._start_signing(now)
        if self.phase is Phase.REVEAL:
            return self._start_countdown(now)
        return []
```

Add this method in the `# ---- messages ----` section, after `_round_start`:

```python
    def _game_over(self) -> dict:
        return {"type": "game_over", "score": self.score, "gallery": list(self.gallery)}
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_game.py -v`
Expected: 15 passed

- [x] **Step 5: Commit**

```bash
git add server/game.py tests/test_game.py
git commit -m "feat: game engine signing, images, scoring and game over"
```

---

### Task 8: Game engine 3/3, timeout, disconnect, reconnect

**Files:**
- Modify: `server/game.py` (add methods, replace `tick`)
- Modify: `tests/test_game.py` (append tests)

- [x] **Step 1: Append the failing tests to `tests/test_game.py`**

```python
def test_timeout_fails_the_round_and_starts_a_new_pair():
    room = lobby()
    start(room)  # signing from t=3, deadline t=13
    sign(room, 1, "RED", 3.0)
    assert room.tick(12.9) == []
    outs = room.tick(13.0)
    assert sent(outs, "round_failed") == [{"type": "round_failed", "round": 1, "reason": "timeout", "slots": [2]}]
    assert room.score == 0 and room.phase is Phase.COUNTDOWN
    room.on_pair("BIG", "DOG")
    outs = room.tick(13.0 + S.countdown)
    assert sent(outs, "round_start", to=1)[0]["your_word"] == "DOG"  # new pair, roles swapped


def test_disconnect_pauses_and_reconnect_resumes_with_the_same_time_left():
    room = lobby()
    start(room)  # deadline t=13
    outs = room.disconnect(2, 5.0)  # 8 s left
    assert sent(outs, "room_state")[0]["paused"] is True
    assert room.tick(20.0) == []  # frozen, no timeout
    assert room.on_recognition(1, onehot("R"), None, 20.0) == []
    outs = room.reconnect(2, 25.0)
    assert sent(outs, "room_joined", to=2)[0]["token"] == "t2"
    assert sent(outs, "round_start", to=2)[0]["your_word"] == "CAT"
    assert [m["slot"] for m in sent(outs, "progress", to=2)] == [1, 2]
    state = sent(outs, "room_state")[0]
    assert state["paused"] is False and state["time_left"] == 8.0
    assert room.tick(32.9) == []
    assert sent(room.tick(33.0), "round_failed")


def test_room_closes_after_the_reconnect_grace():
    room = lobby()
    start(room)
    room.disconnect(2, 5.0)
    assert room.tick(34.9) == []
    outs = room.tick(35.0)
    assert sent(outs, "room_closed") == [{"type": "room_closed", "reason": "partner_left"}]
    assert [o.reason for o in outs if isinstance(o, CloseRoom)] == ["partner_left"]
    assert room.closed and room.tick(40.0) == []


def test_lobby_needs_both_players_connected_to_start():
    room = lobby()
    room.set_ready(2, True, 0.0)
    room.disconnect(2, 0.0)
    room.set_ready(1, True, 0.0)
    assert room.phase is Phase.LOBBY


def test_reconnect_after_game_over_resends_the_gallery():
    room = lobby()
    room.phase = Phase.GAME_OVER
    room.disconnect(1, 0.0)
    assert sent(room.reconnect(1, 1.0), "game_over", to=1)
```

- [x] **Step 2: Run the tests to see the new ones fail**

Run: `uv run pytest tests/test_game.py -v`
Expected: the 5 new tests fail (`tick(13.0)` returns `[]`, or `AttributeError: ... 'disconnect'`); the 15 old tests pass.

- [x] **Step 3: Add the methods to `Room` in `server/game.py`**

Insert this block after `play_again`:

```python
    # ---- connections (Q8) ----

    def disconnect(self, slot: int, now: float) -> list:
        p = self.players.get(slot)
        if p is None or not p.connected or self.closed:
            return []
        p.connected = False
        p.hold.reset()
        if self.paused_at is None:
            self.paused_at = now
            self.close_at = now + self.s.reconnect_grace
        return [self._state(now)]

    def reconnect(self, slot: int, now: float) -> list:
        self.players[slot].connected = True
        if self.paused_at is not None and all(p.connected for p in self.players.values()):
            if self.deadline is not None:
                self.deadline += now - self.paused_at  # timers resume where they froze
            self.paused_at = self.close_at = None
        outs = [Send(slot, self._joined(slot)), self._state(now)]
        if self.phase is Phase.SIGNING:
            outs.append(Send(slot, self._round_start(slot)))
            outs += [Send(slot, {"type": "progress", "slot": p.slot, "done": p.index, "length": len(p.word)})
                     for p in self._ordered()]
        elif self.phase is Phase.GAME_OVER:
            outs.append(Send(slot, self._game_over()))
        return outs
```

Replace the whole `tick` method with:

```python
    def tick(self, now: float) -> list:
        if self.closed:
            return []
        if self.paused_at is not None:  # timers freeze while someone is reconnecting (Q8)
            if now < self.close_at:
                return []
            self.closed = True
            return [Send(None, {"type": "room_closed", "reason": "partner_left"}), CloseRoom("partner_left")]
        if self.deadline is None or now < self.deadline:
            return []
        if self.phase is Phase.COUNTDOWN:
            return self._start_signing(now)
        if self.phase is Phase.SIGNING:
            return self._round_failed(now)
        if self.phase is Phase.REVEAL:
            return self._start_countdown(now)
        return []
```

Add this method in the `# ---- transitions ----` section, after `_start_signing`:

```python
    def _round_failed(self, now: float) -> list:
        missed = [p.slot for p in self._ordered() if not p.done]
        return [Send(None, {"type": "round_failed", "round": self.round_no, "reason": "timeout", "slots": missed}),
                *self._start_countdown(now)]  # 0 points; the prefetched pair starts next (Q13)
```

- [x] **Step 4: Run all tests to see them pass**

Run: `uv run pytest -v`
Expected: all pass (20 in `test_game.py`, plus the earlier suites).

- [x] **Step 5: Commit**

```bash
git add server/game.py tests/test_game.py
git commit -m "feat: game engine timeouts, disconnect pause and reconnect"
```

---

### Task 9: Muse client and placeholder image

**Files:**
- Create: `server/muse.py`, `tests/test_muse.py`

- [x] **Step 1: Write the failing tests**

`tests/test_muse.py`:

```python
import asyncio
import base64
import json

import httpx

from server import muse

BASE = "https://api.meta.ai/v1"


def call(handler, fn, *args):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE) as client:
            return await fn(client, *args)
    return asyncio.run(go())


def test_fetch_pair_sends_structured_output_request_and_parses_it():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        content = json.dumps({"adjective": "red", "noun": "cat"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    assert call(handler, muse.fetch_pair, {"BIG"}) == ("red", "cat")
    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["body"]["model"] == "muse-spark-1.3"
    fmt = seen["body"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert "BIG" in seen["body"]["messages"][0]["content"]


def test_fetch_pair_returns_none_on_errors():
    assert call(lambda r: httpx.Response(500), muse.fetch_pair, set()) is None
    assert call(lambda r: httpx.Response(200, json={"choices": []}), muse.fetch_pair, set()) is None


def test_generate_image_uses_low_reasoning_without_search():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"img").decode()}]})

    assert call(handler, muse.generate_image, "RED CAT") == (b"img", "image/jpeg")
    assert seen["url"] == f"{BASE}/images/generations"
    body = seen["body"]
    assert body["model"] == "muse-image-1.0"
    assert body["reasoning_strength"] == "low"
    assert body["tool_enablement"] == {"enable_web_search": False, "enable_image_search": False}
    assert body["response_format"] == "b64_json" and body["output_format"] == "jpeg"
    assert "red cat" in body["prompt"]


def test_generate_image_returns_none_on_errors():
    assert call(lambda r: httpx.Response(400, json={"error": "bad"}), muse.generate_image, "RED CAT") is None


def test_placeholder_shows_the_phrase():
    svg = muse.placeholder_svg("RED CAT").decode()
    assert svg.startswith("<svg") and "RED CAT" in svg
    assert "&lt;b&gt;" in muse.placeholder_svg("<b>").decode()
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_muse.py -v`
Expected: collection error, `cannot import name 'muse'`

- [x] **Step 3: Implement `server/muse.py`**

```python
"""Muse (Meta Model API) calls. Each returns None on failure so the caller falls back (Q7, Q19)."""
import base64
import json
import logging
from html import escape

import httpx

log = logging.getLogger(__name__)

TEXT_MODEL = "muse-spark-1.3"  # not a -contributor variant, so Meta does not train on our prompts
IMAGE_MODEL = "muse-image-1.0"

PAIR_SCHEMA = {
    "name": "word_pair",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"adjective": {"type": "string"}, "noun": {"type": "string"}},
        "required": ["adjective", "noun"],
        "additionalProperties": False,
    },
}


def make_client(api_key: str, base_url: str, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)


async def fetch_pair(client: httpx.AsyncClient, used) -> tuple[str, str] | None:
    avoid = ", ".join(sorted(used)) or "none"
    prompt = (
        "Pick one common English adjective and one common concrete noun that together make a fun, "
        "easy-to-draw phrase, like RED CAT. Each word must be 3 to 5 letters long, use only the letters A-Z, "
        f"and must not contain the letters J or Z. Do not use any of these words: {avoid}."
    )
    try:
        r = await client.post("/chat/completions", json={
            "model": TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": PAIR_SCHEMA},
        })
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        return data["adjective"], data["noun"]
    except Exception:  # any failure falls back to the built-in words
        log.exception("Muse word pair failed")
        return None


async def generate_image(client: httpx.AsyncClient, phrase: str) -> tuple[bytes, str] | None:
    try:
        r = await client.post("/images/generations", json={
            "model": IMAGE_MODEL,
            "prompt": f"A bright, friendly cartoon illustration of a {phrase.lower()}, simple background, no text.",
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
            "output_format": "jpeg",
            "reasoning_strength": "low",  # faster (Q19)
            "tool_enablement": {"enable_web_search": False, "enable_image_search": False},
        })
        r.raise_for_status()
        return base64.b64decode(r.json()["data"][0]["b64_json"]), "image/jpeg"
    except Exception:  # any failure shows the placeholder instead
        log.exception("Muse image failed")
        return None


def placeholder_svg(phrase: str) -> bytes:
    """Shown when Muse is off, fails, or takes longer than the image timeout (Q19)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">'
        '<rect width="1024" height="1024" fill="#f4efe6"/>'
        '<text x="512" y="540" font-family="sans-serif" font-size="96" font-weight="700" '
        f'text-anchor="middle" fill="#333">{escape(phrase)}</text></svg>'
    ).encode()
```

- [x] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_muse.py -v`
Expected: 5 passed (logged tracebacks from the error tests are expected).

- [x] **Step 5: Commit**

```bash
git add server/muse.py tests/test_muse.py
git commit -m "feat: Muse word-pair and image client with placeholder image"
```

---

### Task 10: FastAPI server and WebSocket protocol

**Files:**
- Create: `server/app.py`, `server/main.py`, `tests/test_ws.py`

- [x] **Step 1: Write the failing tests**

`tests/test_ws.py`:

```python
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from server.app import create_app
from server.classifier import LETTERS
from server.config import Settings

FAST = Settings(countdown=0.2, hold_seconds=0.05, reveal=0.2, api_key=None)


class LetterStub:
    """Test-only recognizer: each frame's bytes name the letter it shows."""

    def process(self, jpeg):
        probs = np.zeros(26, dtype=np.float32)
        probs[LETTERS.index(jpeg.decode())] = 1.0
        return SimpleNamespace(probs=probs, landmarks=None)

    def close(self):
        pass


def recv(ws, kind):
    while True:
        msg = ws.receive_json()
        if msg["type"] == kind:
            return msg


def sign(ws, word):
    for letter in word:
        while True:
            ws.send_bytes(letter.encode())
            if recv(ws, "recognition")["accepted"]:
                break


def test_full_round_over_the_websocket():
    app = create_app(FAST, make_recognizer=LetterStub)
    with TestClient(app) as client, client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.send_json({"type": "create_room", "name": "Ann"})
        joined = recv(a, "room_joined")
        assert joined["slot"] == 1 and joined["token"]
        b.send_json({"type": "join_room", "room": joined["room"].lower(), "name": "Ben"})
        assert recv(b, "room_joined")["slot"] == 2
        a.send_json({"type": "ready", "ready": True})
        b.send_json({"type": "ready", "ready": True})
        word_a = recv(a, "round_start")["your_word"]
        word_b = recv(b, "round_start")["your_word"]
        sign(a, word_a)
        sign(b, word_b)
        ready = recv(a, "image_ready")
        assert ready["phrase"] == f"{word_a} {word_b}"  # slot 1 has the adjective in round 1
        assert recv(a, "round_complete")["score"] == 100
        image = client.get(ready["url"])
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/svg+xml")
        assert client.get("/healthz").json() == {"ok": True, "rooms": 1}


def test_join_unknown_room_errors():
    app = create_app(FAST, make_recognizer=LetterStub)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "join_room", "room": "QQQQ", "name": "X"})
        assert ws.receive_json()["code"] == "room_not_found"


def test_first_message_must_join():
    app = create_app(FAST, make_recognizer=LetterStub)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ready"})
        assert ws.receive_json()["code"] == "join_first"


def test_rejoin_with_token_restores_the_slot():
    app = create_app(FAST, make_recognizer=LetterStub)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as a:
            a.send_json({"type": "create_room", "name": "Ann"})
            joined = recv(a, "room_joined")
        with client.websocket_connect("/ws") as again:
            again.send_json({"type": "rejoin", "room": joined["room"], "token": joined["token"]})
            assert recv(again, "room_joined")["slot"] == 1
            assert recv(again, "room_state")["players"][0]["connected"] is True
        with client.websocket_connect("/ws") as bad:
            bad.send_json({"type": "rejoin", "room": joined["room"], "token": "wrong"})
            assert bad.receive_json()["code"] == "bad_token"


def test_unknown_image_is_404():
    with TestClient(create_app(FAST, make_recognizer=LetterStub)) as client:
        assert client.get("/images/QQQQ/1").status_code == 404
```

- [x] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_ws.py -v`
Expected: collection error, `No module named 'server.app'`

- [x] **Step 3: Implement `server/app.py`**

```python
"""HTTP and WebSocket server around the pure game engine in server/game.py."""
import asyncio
import json
import logging
import secrets
import string
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import muse
from .config import MODELS, Settings
from .game import CloseRoom, FetchPair, GenerateImage, Phase, Room, Send

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
TICK = 0.1       # room timer resolution, seconds
FRAME_GAP = 0.1  # at most 10 processed frames per second per player (Q16)


def error_msg(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message}


@dataclass(eq=False)
class Conn:
    slot: int
    ws: WebSocket
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue)  # None closes the socket
    frame: bytes | None = None
    frame_ready: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(eq=False)
class Live:
    room: Room
    conns: dict = field(default_factory=dict)   # slot -> Conn
    images: dict = field(default_factory=dict)  # image id -> (bytes, content type)
    tasks: set = field(default_factory=set)


class Hub:
    """Owns rooms and all I/O around them: sockets, recognizer threads, timers, Muse calls."""

    def __init__(self, settings: Settings, make_recognizer):
        self.s = settings
        self.make_recognizer = make_recognizer
        self.rooms: dict[str, Live] = {}
        self.muse = None  # httpx.AsyncClient when MODEL_API_KEY is set

    # ---- plumbing ----

    def spawn(self, live: Live, coro):
        task = asyncio.create_task(coro)
        live.tasks.add(task)
        task.add_done_callback(live.tasks.discard)

    def dispatch(self, live: Live, outs: list):
        for o in outs:
            if isinstance(o, Send):
                for slot, conn in live.conns.items():
                    if o.to in (None, slot):
                        conn.outbox.put_nowait(o.msg)
            elif isinstance(o, FetchPair):
                self.spawn(live, self.fetch_pair(live, o.used))
            elif isinstance(o, GenerateImage):
                self.spawn(live, self.generate(live, o))
            elif isinstance(o, CloseRoom):
                self.close(live)

    def close(self, live: Live):
        self.rooms.pop(live.room.code, None)
        for conn in live.conns.values():
            conn.outbox.put_nowait(None)
        for task in list(live.tasks):
            task.cancel()

    def new_code(self) -> str:
        while True:
            code = "".join(secrets.choice(string.ascii_uppercase) for _ in range(4))
            if code not in self.rooms:
                return code

    # ---- background work ----

    async def ticker(self, live: Live):
        while not live.room.closed:
            await asyncio.sleep(TICK)
            self.dispatch(live, live.room.tick(time.monotonic()))

    async def fetch_pair(self, live: Live, used):
        if self.muse is None:
            return  # the engine falls back to the built-in words
        try:
            pair = await asyncio.wait_for(muse.fetch_pair(self.muse, used), self.s.word_timeout)
        except TimeoutError:
            pair = None
        if pair:
            self.dispatch(live, live.room.on_pair(*pair))

    async def generate(self, live: Live, job: GenerateImage):
        image = None
        if self.muse is not None:
            try:
                image = await asyncio.wait_for(muse.generate_image(self.muse, job.phrase), self.s.image_timeout)
            except TimeoutError:
                log.warning("image for %s took over %s s", job.phrase, self.s.image_timeout)
        image = image or (muse.placeholder_svg(job.phrase), "image/svg+xml")
        image_id = str(len(live.images) + 1)
        live.images[image_id] = image
        url = f"/images/{live.room.code}/{image_id}"
        self.dispatch(live, live.room.on_image(job.round_no, url, time.monotonic()))

    async def sender(self, conn: Conn):
        try:
            while (msg := await conn.outbox.get()) is not None:
                await conn.ws.send_json(msg)
            await conn.ws.close()
        except Exception:  # the socket is already gone; the receive loop cleans up
            pass

    async def recognize(self, live: Live, conn: Conn):
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1)  # one thread per player keeps MediaPipe calls serial
        rec = None
        try:
            rec = await loop.run_in_executor(pool, self.make_recognizer)
            while True:
                await conn.frame_ready.wait()
                conn.frame_ready.clear()
                jpeg, conn.frame = conn.frame, None
                started = time.monotonic()
                result = await loop.run_in_executor(pool, rec.process, jpeg)
                if result is None:
                    conn.outbox.put_nowait(error_msg("bad_frame", "Frames must be JPEG images of at most 640x480."))
                else:
                    outs = live.room.on_recognition(conn.slot, result.probs, result.landmarks, time.monotonic())
                    self.dispatch(live, outs)
                await asyncio.sleep(max(0.0, FRAME_GAP - (time.monotonic() - started)))
        finally:
            if rec is not None:
                pool.submit(rec.close)  # queued behind any frame still being processed
            pool.shutdown(wait=False)

    # ---- connections ----

    def join(self, m: dict, ws: WebSocket):
        """Returns (live, conn, outputs), or an error message."""
        now = time.monotonic()
        kind = m.get("type")
        name = str(m.get("name") or "Player")[:20]
        if kind == "create_room":
            live = Live(Room(self.new_code(), self.s))
            self.rooms[live.room.code] = live
            self.spawn(live, self.ticker(live))
            slot, outs = live.room.join(name, secrets.token_urlsafe(16), now)
        elif kind in ("join_room", "rejoin"):
            live = self.rooms.get(str(m.get("room", "")).strip().upper())
            if live is None:
                return error_msg("room_not_found", "No room with that code.")
            if kind == "join_room":
                if live.room.full:
                    return error_msg("room_full", "That room already has two players.")
                slot, outs = live.room.join(name, secrets.token_urlsafe(16), now)
            else:
                slot = live.room.slot_for_token(str(m.get("token", "")))
                if slot is None:
                    return error_msg("bad_token", "That player token is not valid for this room.")
                if slot in live.conns:
                    live.conns[slot].outbox.put_nowait(None)  # a newer connection replaces the old one
                outs = live.room.reconnect(slot, now)
        else:
            return error_msg("join_first", "Send create_room, join_room or rejoin first.")
        conn = Conn(slot, ws)
        live.conns[slot] = conn
        return live, conn, outs

    def on_frame(self, live: Live, conn: Conn, data: bytes):
        if live.room.phase is not Phase.SIGNING:
            return
        if len(data) > self.s.max_frame_bytes:
            conn.outbox.put_nowait(error_msg("frame_too_large", f"Frames must be under {self.s.max_frame_bytes} bytes."))
            return
        conn.frame = data  # keep only the newest frame (Q16)
        conn.frame_ready.set()

    def on_text(self, live: Live, conn: Conn, text: str):
        try:
            m = json.loads(text)
        except ValueError:
            m = None
        if not isinstance(m, dict):
            conn.outbox.put_nowait(error_msg("bad_message", "Messages must be JSON objects."))
            return
        now = time.monotonic()
        kind = m.get("type")
        if kind == "ready":
            self.dispatch(live, live.room.set_ready(conn.slot, bool(m.get("ready", True)), now))
        elif kind == "play_again":
            self.dispatch(live, live.room.play_again(conn.slot, now))
        else:
            conn.outbox.put_nowait(error_msg("unknown_type", f"Unknown message type {kind!r}."))

    async def serve(self, ws: WebSocket):
        await ws.accept()
        try:
            first = json.loads(await ws.receive_text())
        except WebSocketDisconnect:
            return
        except (ValueError, KeyError):  # not JSON, or a binary first message
            first = None
        joined = self.join(first, ws) if isinstance(first, dict) else error_msg("join_first", "Send create_room, join_room or rejoin first.")
        if isinstance(joined, dict):
            await ws.send_json(joined)
            await ws.close()
            return
        live, conn, outs = joined
        sender = asyncio.create_task(self.sender(conn))
        worker = asyncio.create_task(self.recognize(live, conn))
        self.dispatch(live, outs)
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    self.on_frame(live, conn, msg["bytes"])
                elif msg.get("text") is not None:
                    self.on_text(live, conn, msg["text"])
        finally:
            worker.cancel()
            sender.cancel()
            if live.conns.get(conn.slot) is conn and not live.room.closed:
                del live.conns[conn.slot]
                self.dispatch(live, live.room.disconnect(conn.slot, time.monotonic()))


def create_app(settings: Settings | None = None, make_recognizer=None) -> FastAPI:
    settings = settings or Settings()
    if make_recognizer is None:
        from .classifier import Classifier
        from .recognizer import Recognizer  # imported here so tests with a stub skip MediaPipe

        classifier = Classifier(MODELS / "classifier.npz")

        def make_recognizer():
            return Recognizer(classifier, max_w=settings.max_frame_width, max_h=settings.max_frame_height)

    hub = Hub(settings, make_recognizer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.api_key:
            hub.muse = muse.make_client(settings.api_key, settings.muse_base_url, settings.image_timeout)
        yield
        if hub.muse is not None:
            await hub.muse.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub
    # The UI may be hosted anywhere (Q22). WebSockets ignore CORS; this covers fetch() calls.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "rooms": len(hub.rooms)}

    @app.get("/images/{code}/{image_id}")
    def image(code: str, image_id: str):
        live = hub.rooms.get(code)
        found = live.images.get(image_id) if live else None
        if found is None:
            return Response(status_code=404)
        return Response(found[0], media_type=found[1])

    @app.get("/dev")
    def dev():
        return FileResponse(STATIC / "dev.html")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await hub.serve(websocket)

    return app
```

- [x] **Step 4: Write `server/main.py`**

```python
import logging

from .app import create_app

logging.basicConfig(level=logging.INFO)
app = create_app()
```

- [x] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_ws.py -v`
Expected: 5 passed, in under 15 seconds.

- [x] **Step 6: Check the real app starts**

Run: `timeout 20 uv run uvicorn server.main:app --port 8765 & sleep 8; curl -s localhost:8765/healthz; wait`
Expected: `{"ok":true,"rooms":0}` printed before the timeout stops the server.

- [x] **Step 7: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add server/app.py server/main.py tests/test_ws.py
git commit -m "feat: FastAPI WebSocket server, room registry and image endpoint"
```

---

### Task 11: Protocol document for the frontend teammate

**Files:**
- Create: `docs/protocol.md`

- [x] **Step 1: Write `docs/protocol.md`**

````markdown
# ASL Co-op WebSocket Protocol

This is the contract between the game UI and the backend. The backend implements exactly this.

## Endpoints

| What | Production | Local |
|---|---|---|
| WebSocket | `wss://<app>.fly.dev/ws` | `ws://localhost:8000/ws` |
| Images | `https://<app>.fly.dev` + the `url` in `image_ready` | `http://localhost:8000` + `url` |
| Health | `GET /healthz` returns `{"ok": true, "rooms": N}` | same |
| Test page | `GET /dev` | same |

Image URLs are paths such as `/images/ABCD/1`. Resolve them against the backend origin: `new URL(msg.url, BACKEND_ORIGIN)`. `<img>` tags and `fetch()` work from any site.

## Game flow

1. Open the socket. The first message must be `create_room`, `join_room`, or `rejoin`.
2. The server answers with `room_joined`. Store `room` and `token`.
3. Both players send `ready`. A 3-second `countdown` starts.
4. `round_start` gives each player a word. The phase becomes `signing`, and the team has 10 seconds in total to finish both words.
5. While signing, send camera frames as binary messages. You get a `recognition` for each processed frame, and both players get a `progress` each time a letter is accepted.
6. When both words are done, everyone gets `image_pending`, then `image_ready` and `round_complete` (+100 points). The image shows for 5 seconds, then the next countdown starts.
7. On timeout, everyone gets `round_failed` (0 points) and a new countdown with a fresh pair starts.
8. At 500 points, everyone gets `game_over` with the gallery. Either player can send `play_again` to return to the lobby with the score reset.

Phases, as sent in `room_state.phase`: `lobby`, `countdown`, `signing`, `generating`, `reveal`, `game_over`.

## Camera frames (client to server, binary)

- One JPEG image per binary WebSocket message.
- Send the **raw, unmirrored** camera image. Mirror only your on-screen preview (CSS `transform: scaleX(-1)`).
- At most 640x480 pixels, under 300 KB, at most 10 per second. JPEG quality 0.7 is plenty.
- Send frames only while `phase` is `signing` and `paused` is false. The server ignores other frames.
- The server keeps only the newest frame per player, so a slow connection never builds a queue.

```js
canvas.width = 640; canvas.height = 480;
canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
canvas.toBlob((blob) => ws.send(blob), "image/jpeg", 0.7);
```

## Client to server (JSON text)

| Message | When |
|---|---|
| `{"type":"create_room","name":"Ann"}` | First message, to start a new room. Name up to 20 characters. |
| `{"type":"join_room","room":"ABCD","name":"Ben"}` | First message, to join a friend's room. The code is not case-sensitive. |
| `{"type":"rejoin","room":"ABCD","token":"..."}` | First message after a dropped connection. Allowed within 30 seconds. |
| `{"type":"ready","ready":true}` | In the lobby. Send `false` to un-ready. |
| `{"type":"play_again"}` | After `game_over`. |

## Server to client (JSON text)

`room_joined`, sent only to you:
```json
{"type":"room_joined","room":"ABCD","slot":1,"token":"x7..."}
```
Keep `token` for `rejoin`. `slot` is 1 or 2.

`room_state`, sent to both players after every state change:
```json
{"type":"room_state","room":"ABCD","phase":"signing","paused":false,"score":100,"target":500,"round":2,
 "time_left":10.0,
 "players":[{"slot":1,"name":"Ann","ready":true,"connected":true},
            {"slot":2,"name":"Ben","ready":true,"connected":true}]}
```
`time_left` is the number of seconds left in the round during `signing`, and `null` otherwise. Restart your countdown display from it each time this message arrives. `paused` is true while a player is reconnecting. Timers freeze until the player returns.

`countdown`:
```json
{"type":"countdown","round":2,"seconds":3.0}
```

`round_start`, sent to each player with their own `your_word`:
```json
{"type":"round_start","round":2,"phrase":"BIG DOG","words":[{"slot":1,"word":"DOG"},{"slot":2,"word":"BIG"}],
 "your_word":"DOG","time_limit":10.0}
```
Words are 3 to 5 letters, A to Z, never J or Z. Players swap between the adjective and the noun each round.

`recognition`, sent only to the signer, once per processed frame (up to 10 per second):
```json
{"type":"recognition","guess":"D","confidence":0.93,"hold":0.5,"accepted":false,"expected":"D",
 "landmarks":[[0.41,0.62],[0.45,0.58], ...]}
```
- `guess` is the model's top letter, or `null` when no hand is seen. `confidence` runs from 0 to 1.
- A letter is accepted when `guess` equals `expected` with confidence of at least 0.8, held for 0.6 seconds. `hold` is that hold's progress, from 0 to 1. Wrong letters carry no penalty.
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
{"type":"image_ready","round":2,"phrase":"BIG DOG","url":"/images/ABCD/2"}
{"type":"round_complete","round":2,"phrase":"BIG DOG","points":100,"score":200}
```
The image is a Muse-generated JPEG. If Muse is not configured, fails, or takes more than 10 seconds, it is an SVG placeholder that shows the phrase. Images last as long as the room.

`game_over`:
```json
{"type":"game_over","score":500,"gallery":[{"phrase":"RED CAT","url":"/images/ABCD/1"}, ...]}
```

`room_closed`, sent when a player stays disconnected for 30 seconds. The room is gone, so create a new one.
```json
{"type":"room_closed","reason":"partner_left"}
```

`error`:
```json
{"type":"error","code":"room_not_found","message":"No room with that code."}
```

| Code | Meaning | Socket |
|---|---|---|
| `join_first` | The first message was not create, join, or rejoin | closed |
| `room_not_found` | Wrong code, or the room has closed | closed |
| `room_full` | The room already has two players | closed |
| `bad_token` | Unknown token for this room | closed |
| `bad_message` | The text was not a JSON object | stays open |
| `unknown_type` | Unknown `type` | stays open |
| `frame_too_large` | Frame of 300 KB or more | stays open |
| `bad_frame` | Not a JPEG, or larger than 640x480 | stays open |

## Reconnecting

When the socket closes unexpectedly, open a new one and send `rejoin` with the stored `room` and `token`. A newer connection replaces an older one for the same player. After `room_joined`, you get the current `room_state`. During `signing` you also get `round_start` and both players' `progress`, and during `game_over` you get `game_over`.

## Running the backend locally

```bash
uv sync
uv run uvicorn server.main:app --port 8000
# optional: real words and images
MODEL_API_KEY=... uv run uvicorn server.main:app --port 8000
```

Browsers allow the camera only on HTTPS or `localhost`. If the backend runs on a remote machine, tunnel it with `ssh -L 8000:localhost:8000 <server>` and open `http://localhost:8000/dev`.
````

- [x] **Step 2: Check the doc against the code**

Run: `grep -o '"type": "[a-z_]*"' server/game.py server/app.py | sort -u`
Expected: every message type the code sends (`countdown`, `error`, `game_over`, `image_pending`, `image_ready`, `progress`, `recognition`, `room_closed`, `room_joined`, `room_state`, `round_complete`, `round_failed`, `round_start`) appears in `docs/protocol.md`. Fix the doc if any field differs from the code. `error` is built by `error_msg()` in `server/app.py`, so grep may show it as `"type": "error"` there.

- [x] **Step 3: Commit**

```bash
git add docs/protocol.md
git commit -m "docs: WebSocket protocol for the frontend"
```

---

### Task 12: /dev test page

**Files:**
- Create: `server/static/dev.html`

- [x] **Step 1: Write `server/static/dev.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASL Co-op Dev</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; max-width: 960px; }
  button, input { font-size: 16px; }
  #stage { position: relative; width: 640px; max-width: 100%; aspect-ratio: 4 / 3; background: #111; }
  #stage video, #stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #stage video { transform: scaleX(-1); object-fit: fill; } /* mirror the preview only (Q16) */
  .word { font-size: 32px; letter-spacing: 6px; font-family: monospace; }
  .done { color: #1a7f1a; }
  .next { text-decoration: underline; }
  #hold { width: 240px; height: 10px; background: #ddd; }
  #hold div { height: 100%; width: 0; background: #1a7f1a; }
  #image img, #gallery img { width: 256px; margin: 4px; }
  pre { background: #f4f4f4; max-height: 200px; overflow: auto; font-size: 12px; }
</style>
</head>
<body>
<h1>ASL Co-op dev page</h1>
<p>
  <label>Name <input id="name" value="Player" maxlength="20"></label>
  <button id="create">Create room</button>
  <label>Room <input id="code" size="5" maxlength="4"></label>
  <button id="join">Join room</button>
  <button id="ready" disabled>Ready</button>
  <button id="again" hidden>Play again</button>
</p>
<p id="status">Not connected</p>
<div id="stage">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay" width="640" height="480"></canvas>
</div>
<p>Guess: <b id="guess">-</b> <span id="conf"></span></p>
<div id="hold"><div></div></div>
<p>You: <span id="mine" class="word"></span></p>
<p>Partner: <span id="theirs" class="word"></span></p>
<p>Time left: <b id="timer">-</b> · Score: <b id="score">0</b> / <span id="target">500</span></p>
<div id="image"></div>
<div id="gallery"></div>
<pre id="log"></pre>
<script>
const $ = (id) => document.getElementById(id);
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
               [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
const video = $("video"), overlay = $("overlay").getContext("2d"), grab = document.createElement("canvas");
let ws = null, session = null, me = 0, phase = "", paused = false, ready = false;
let words = {}, progress = {}, endsAt = 0, encoding = false;

function log(m) { $("log").textContent = JSON.stringify(m) + "\n" + $("log").textContent.slice(0, 5000); }

function send(m) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m)); }

function connect(first) {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => ws.send(JSON.stringify(first));
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    $("status").textContent = "Disconnected";
    // Rejoin with the player token while the server holds the slot (30 s, Q8)
    if (session) setTimeout(() => connect({ type: "rejoin", room: session.room, token: session.token }), 2000);
  };
}

async function startCamera() {
  if (!video.srcObject) video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
}

function img(src, alt) { const i = new Image(); i.src = src; i.alt = alt; i.title = alt; return i; }

function letters(word, done) {
  return [...word].map((c, i) => `<span class="${i < done ? "done" : i === done ? "next" : ""}">${c}</span>`).join("");
}

function drawWords() {
  const other = me === 1 ? 2 : 1;
  $("mine").innerHTML = letters(words[me] || "", progress[me] || 0);
  $("theirs").innerHTML = letters(words[other] || "", progress[other] || 0);
}

function drawRecognition(m) {
  $("guess").textContent = m.guess || "-";
  $("conf").textContent = m.guess ? `${Math.round(m.confidence * 100)}% (need ${m.expected})` : "no hand";
  $("hold").firstElementChild.style.width = `${m.hold * 100}%`;
  overlay.clearRect(0, 0, 640, 480);
  if (!m.landmarks) return;
  const pts = m.landmarks.map(([x, y]) => [x * 640, y * 480]); // already mirrored, like the preview
  overlay.strokeStyle = "#0c0"; overlay.fillStyle = "#0f0"; overlay.lineWidth = 2;
  for (const [a, b] of BONES) { overlay.beginPath(); overlay.moveTo(...pts[a]); overlay.lineTo(...pts[b]); overlay.stroke(); }
  for (const [x, y] of pts) { overlay.beginPath(); overlay.arc(x, y, 4, 0, 2 * Math.PI); overlay.fill(); }
}

function handle(m) {
  if (m.type !== "recognition") log(m);
  switch (m.type) {
    case "room_joined":
      me = m.slot; session = { room: m.room, token: m.token };
      $("code").value = m.room; $("ready").disabled = false;
      break;
    case "room_state": {
      phase = m.phase; paused = m.paused;
      ready = !!m.players.find((p) => p.slot === me)?.ready;
      endsAt = m.time_left == null ? 0 : performance.now() + m.time_left * 1000;
      $("ready").textContent = ready ? "Unready" : "Ready";
      $("score").textContent = m.score; $("target").textContent = m.target;
      $("again").hidden = phase !== "game_over";
      const who = m.players.map((p) => `${p.slot === me ? "you" : p.name}${p.ready ? " (ready)" : ""}${p.connected ? "" : " (offline)"}`);
      $("status").textContent = `Room ${m.room} · ${phase}${paused ? " · paused, waiting for a reconnect" : ""} · ${who.join(", ")}`;
      if (phase !== "signing") overlay.clearRect(0, 0, 640, 480);
      break;
    }
    case "countdown": $("image").replaceChildren(); $("timer").textContent = `starting in ${m.seconds}s`; break;
    case "round_start":
      words = {}; progress = {};
      for (const w of m.words) { words[w.slot] = w.word; progress[w.slot] = 0; }
      drawWords();
      break;
    case "progress": progress[m.slot] = m.done; drawWords(); break;
    case "recognition": drawRecognition(m); break;
    case "round_failed": $("timer").textContent = "time up, no points"; break;
    case "image_pending": $("image").textContent = `Drawing ${m.phrase}...`; break;
    case "image_ready": $("image").replaceChildren(img(m.url, m.phrase)); break;
    case "round_complete": $("score").textContent = m.score; break;
    case "game_over": $("gallery").replaceChildren(...m.gallery.map((g) => img(g.url, g.phrase))); break;
    case "room_closed": session = null; break;
    case "error": if (["room_not_found", "room_full", "bad_token"].includes(m.code)) session = null; break;
  }
}

// Raw, unmirrored frames, at most 640x480 and 10 per second, only while signing (Q16)
setInterval(() => {
  if (phase !== "signing" || paused || encoding || !ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  const s = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
  grab.width = Math.round(video.videoWidth * s); grab.height = Math.round(video.videoHeight * s);
  grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
  encoding = true;
  grab.toBlob((b) => { encoding = false; if (b && ws.readyState === WebSocket.OPEN) ws.send(b); }, "image/jpeg", 0.7);
}, 100);

setInterval(() => {
  if (phase === "signing" && endsAt && !paused) $("timer").textContent = `${Math.max(0, (endsAt - performance.now()) / 1000).toFixed(1)}s`;
}, 100);

$("create").onclick = async () => { await startCamera(); connect({ type: "create_room", name: $("name").value }); };
$("join").onclick = async () => {
  await startCamera();
  connect({ type: "join_room", room: $("code").value.trim().toUpperCase(), name: $("name").value });
};
$("ready").onclick = () => send({ type: "ready", ready: !ready });
$("again").onclick = () => { $("gallery").replaceChildren(); send({ type: "play_again" }); };
</script>
</body>
</html>
```

- [x] **Step 2: Check the script parses, if node is available**

Run: `command -v node && sed -n '/<script>/,/<\/script>/p' server/static/dev.html | sed '1d;$d' > /tmp/devcheck.js && node --check /tmp/devcheck.js && echo syntax-ok`
Expected: `syntax-ok`. If node is missing, say so in the report and skip this step.

- [x] **Step 3: Check the server serves it**

Run: `timeout 20 uv run uvicorn server.main:app --port 8765 & sleep 8; curl -s localhost:8765/dev | grep -c "ASL Co-op dev page"; wait`
Expected: `1`

- [x] **Step 4: Commit**

```bash
git add server/static/dev.html
git commit -m "feat: minimal /dev camera test page"
```

---

### Task 13: Docker image and fly.io config

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `fly.toml`

- [x] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

# opencv-contrib-python (pulled in by mediapipe) needs these shared libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.4

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev
COPY server server
COPY models models

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
```

- [x] **Step 2: Write `.dockerignore`**

```
.git
.venv
.cache
.pytest_cache
__pycache__
tests
scripts
docs
handoff.md
Research ASL Models.html
.env
```

- [x] **Step 3: Write `fly.toml`**

```toml
# One machine, always on: rooms live in memory (Q2, Q21)
app = "asl-coop-backend"  # change if the name is taken
primary_region = "dfw"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "off"
  auto_start_machines = true
  min_machines_running = 1

  [[http_service.checks]]
    grace_period = "30s"
    interval = "15s"
    timeout = "5s"
    method = "GET"
    path = "/healthz"

[[vm]]
  size = "shared-cpu-2x"
  memory = "2gb"
```

- [x] **Step 4: Build the image**

Run: `docker build -t asl-coop:dev .`
Expected: the build finishes with no errors.

- [x] **Step 5: Run the container and check health and recognition**

```bash
docker run -d --rm --name asl-coop-test -p 18080:8080 asl-coop:dev
sleep 10
curl -s localhost:18080/healthz
curl -s -o /dev/null -w "%{http_code}\n" localhost:18080/dev
docker exec asl-coop-test python -c "
import cv2, numpy as np
from server.classifier import Classifier
from server.config import MODELS
from server.recognizer import Recognizer
r = Recognizer(Classifier(MODELS / 'classifier.npz'))
print('no-hand', r.process(cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()))
r.close()
"
docker rm -f asl-coop-test
```

Expected: `{"ok":true,"rooms":0}`, then `200`, then `no-hand Recognition(probs=None, landmarks=None)`. If the import fails with a missing `.so` file, add that library's Debian package to the `apt-get install` line and rebuild.

- [x] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore fly.toml
git commit -m "chore: Dockerfile and fly.io config for one always-on machine"
```

---

## After the tasks (the user does these)

1. **Camera test locally.** On the laptop, run `ssh -L 8000:localhost:8000 <dev server>`. On the server, run `uv run uvicorn server.main:app --port 8000`. Open `http://localhost:8000/dev` in two tabs, create a room in one and join it from the other, ready both, and sign. If letters are hard to hold, tune `min_confidence` or `time_limit` in `server/config.py`.
2. **Deploy.** Install flyctl (`curl -L https://fly.io/install.sh | sh`), run `fly auth login`, then `fly launch --no-deploy --copy-config --name <app>`, `fly secrets set MODEL_API_KEY=<key>`, `fly deploy --ha=false`, and `fly scale count 1`. Check `https://<app>.fly.dev/healthz`, then run the camera test at `https://<app>.fly.dev/dev`.
3. **Share** `docs/protocol.md` and the fly URL with the frontend teammate.
