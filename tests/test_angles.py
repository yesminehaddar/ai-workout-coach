"""Tests de features/angles.py."""
import pytest

from coach.features.angles import FEATURE_NAMES, compute_features, joint_angle
from coach.pose.estimator import Landmark


def test_right_angle():
    assert joint_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90)


def test_straight_line_is_180():
    assert joint_angle((-1, 0), (0, 0), (2, 0)) == pytest.approx(180)


def test_degenerate_returns_zero():
    assert joint_angle((0, 0), (0, 0), (1, 1)) == 0.0


def _standing(vis_left=1.0, vis_right=0.2):
    """Personne debout, vue de profil, tournée vers la droite de l'image."""
    lm = [Landmark(0.5, 0.5, 0, 0.0) for _ in range(33)]
    lm[0] = Landmark(0.55, 0.15, 0, 1.0)  # nez, devant la hanche
    for ids, v in [((11, 13, 15, 23, 25, 27), vis_left), ((12, 14, 16, 24, 26, 28), vis_right)]:
        sh, el, wr, hip, kn, an = ids
        for i, y in zip((sh, el, wr, hip, kn, an), (0.25, 0.40, 0.55, 0.55, 0.75, 0.95)):
            lm[i] = Landmark(0.5, y, 0, v)
    return lm


def test_compute_features_standing():
    f = compute_features(_standing())
    assert set(f) == set(FEATURE_NAMES)
    assert f["knee"] == pytest.approx(180, abs=1)
    assert f["elbow"] == pytest.approx(180, abs=1)
    assert f["torso_lean"] == pytest.approx(0, abs=1)
    assert f["body_line"] == pytest.approx(180, abs=1)
    assert f["side"] == 0.0


def test_picks_more_visible_side():
    f = compute_features(_standing(vis_left=0.1, vis_right=0.9))
    assert f["side"] == 1.0
