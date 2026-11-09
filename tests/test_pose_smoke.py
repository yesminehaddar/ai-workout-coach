"""Test de fumée : vraie vidéo du dataset -> pose -> features."""
from pathlib import Path

import cv2
import pytest

from coach.features.angles import FEATURE_NAMES, compute_features
from coach.pose.estimator import DEFAULT_MODEL, PoseEstimator

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "raw" / "fitness_form" / "subject_001_squat_good_side.mp4"

pytestmark = pytest.mark.skipif(
    not (VIDEO.exists() and DEFAULT_MODEL.exists()), reason="vidéo ou modèle de pose absent"
)


def test_squat_video_gives_sensible_angles():
    cap = cv2.VideoCapture(str(VIDEO))
    est = PoseEstimator()
    knees, n = [], 0
    while n < 150:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        h, w = frame.shape[:2]
        lm = est.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms=int(n * 1000 / 30))
        if lm is None:
            continue
        f = compute_features(lm, aspect=w / h)
        assert set(f) == set(FEATURE_NAMES)
        knees.append(f["knee"])
    cap.release()
    est.close()
    assert len(knees) > 50
    assert max(knees) - min(knees) > 40  # le squat plie vraiment le genou
