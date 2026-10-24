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


def test_decompression_bomb_is_rejected(rec):
    assert rec.process(jpeg(4000, 4000)) is None
