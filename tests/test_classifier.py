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
