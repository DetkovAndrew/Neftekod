import numpy as np
import pytest

from neftekod_mas.optimization.joint_envelope import JointEnvelopeChecker

BOUNDS = {
    "_meta": {},
    "avt:T55": {"p05": 375.0, "p50": 380.0, "p95": 385.0},
    "242000:T5": {"p05": 350.0, "p50": 365.0, "p95": 385.0},
}


def _checker():
    # облако из одной точки в центре нормализованного пространства
    points = np.array([[0.5, 0.5]], dtype="float32")
    return JointEnvelopeChecker(points=points, tags=["avt:T55", "242000:T5"], bounds=BOUNDS)


def test_distance_zero_at_the_reference_point():
    je = _checker()
    # значение, которое нормализуется ровно в (0.5, 0.5)
    d = je.nearest_distance({"avt:T55": 380.0, "242000:T5": 367.5})
    assert d == pytest.approx(0.0, abs=1e-4)


def test_distance_grows_away_from_reference_point():
    je = _checker()
    d_near = je.nearest_distance({"avt:T55": 380.0, "242000:T5": 367.5})
    d_far = je.nearest_distance({"avt:T55": 375.0, "242000:T5": 350.0})
    assert d_far > d_near


def test_missing_tag_returns_none():
    je = _checker()
    assert je.nearest_distance({"avt:T55": 380.0}) is None
