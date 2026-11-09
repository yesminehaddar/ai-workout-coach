"""Tests du filtre One Euro et de la fenêtre glissante."""
import numpy as np

from coach.features.angles import FEATURE_NAMES
from coach.features.smoothing import OneEuroFilter
from coach.features.window import SlidingWindow


def test_constant_signal_unchanged():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = [f(5.0, t / 30) for t in range(60)]
    assert all(abs(v - 5.0) < 1e-9 for v in out)


def test_noise_reduced():
    rng = np.random.default_rng(0)
    noisy = 10 + rng.normal(0, 1, 300)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    smooth = np.array([f(v, i / 30) for i, v in enumerate(noisy)])
    assert smooth[30:].std() < noisy[30:].std() / 2


def test_step_is_tracked():
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    for i in range(30):
        f(0.0, i / 30)
    out = [f(100.0, (30 + i) / 30) for i in range(30)]
    assert out[-1] > 90


def _feat(v):
    return {k: float(v) for k in FEATURE_NAMES}


def test_window_shape_and_fill():
    w = SlidingWindow(30)
    assert w.as_array().shape == (30, len(FEATURE_NAMES))
    for i in range(10):
        w.push(_feat(i))
    assert not w.is_full()
    assert w.as_array().shape == (30, len(FEATURE_NAMES))
    for i in range(10, 40):
        w.push(_feat(i))
    assert w.is_full()
    assert w.as_array()[-1, 0] == 39 and w.as_array()[0, 0] == 10
