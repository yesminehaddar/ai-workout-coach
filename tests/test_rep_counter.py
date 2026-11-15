"""Tests de la machine à états : une rep complète compte, une demi-rep non."""
import numpy as np
import pytest

from coach.features.smoothing import JumpFilter
from coach.reasoning.rep_counter import RepCounter, RepState

SQUAT = {"angle": "knee", "top_above": 160, "bottom_below": 100}


def feed(counter, angles):
    events = []
    for a in angles:
        e = counter.update({"knee": float(a)})
        if e:
            events.append(e)
    return events


def rep(low, n=30):
    """Une répétition lisse 175 -> low -> 175."""
    down = np.linspace(175, low, n)
    return np.concatenate([down, down[::-1]])


def test_full_rep_counts():
    c = RepCounter("squat", SQUAT)
    ev = feed(c, np.concatenate([[175] * 5, rep(70), rep(70), rep(70)]))
    assert c.count == 3
    assert [e["rep_index"] for e in ev] == [1, 2, 3]
    assert ev[0]["min_angle"] == pytest.approx(70)
    assert ev[0]["tempo_s"] > 0


def test_partial_rep_not_counted():
    c = RepCounter("squat", SQUAT)
    # Après des reps profondes, une demi-rep (175 -> 130) reste « partial ».
    ev = feed(c, np.concatenate([[175] * 5, rep(70), rep(70), rep(130), rep(70)]))
    assert c.count == 3
    assert c.partial_count == 1
    assert [e["counted"] for e in ev] == [True, True, False, True]


def test_noise_near_top_is_ignored():
    c = RepCounter("squat", SQUAT)
    ev = feed(c, [175, 158, 162, 154, 165, 175])
    assert ev == [] and c.count == 0 and c.partial_count == 0


def test_no_count_before_first_top():
    """Démarrer en bas du mouvement ne doit pas compter une demi-rep comme une rep."""
    c = RepCounter("squat", SQUAT)
    feed(c, np.linspace(80, 175, 30))
    assert c.count == 0
    assert c.state == RepState.TOP


def test_glitch_frame_does_not_create_rep():
    """Un seul frame aberrant (29°) en position debout ne doit rien compter."""
    c, jf = RepCounter("squat", SQUAT), JumpFilter()
    for a in [175] * 10 + [29] + [175] * 10:
        c.update(jf({"knee": float(a)}))
    assert c.count == 0


def test_top_calibrates_when_never_fully_extended():
    """Bras qui plafonne à 145° alors que le seuil du haut est 150° : on compte quand même."""
    c = RepCounter("pushup", {"angle": "elbow", "top_above": 150, "bottom_below": 100})
    one = np.concatenate([np.linspace(145, 75, 20), np.linspace(75, 145, 20)])
    for i, a in enumerate(np.concatenate([[145] * 10] + [one] * 3)):
        c.update({"elbow": float(a)}, timestamp=i / 15)
    assert c.count == 3
    assert 120 < c.effective_top < 150


def test_jump_filter_accepts_persistent_change():
    jf = JumpFilter(max_jump=40, max_hold=3)
    out = [jf({"knee": v})["knee"] for v in [170, 170, 60, 60, 60, 60, 60]]
    assert out[2] == 170          # premier saut rejeté
    assert out[-1] == 60          # mais accepté s'il persiste


def test_foreshortened_view_still_counts():
    """Pompe de biais : le bras semble aller de 72 à 110° seulement. Seuils relatifs -> comptée."""
    c = RepCounter("pushup", {"angle": "elbow", "top_above": 150, "bottom_below": 100})
    one = np.concatenate([np.linspace(110, 72, 8), np.linspace(72, 110, 8)])
    for i, a in enumerate(np.concatenate([[110] * 10] + [one] * 6)):
        c.update({"elbow": float(a)}, timestamp=i / 15)
    assert c.count >= 5


def test_small_wobble_is_not_a_rep():
    """Moins de min_amplitude (25°) de mouvement : rien n'est compté."""
    c = RepCounter("squat", SQUAT)
    wob = np.concatenate([np.linspace(178, 160, 10), np.linspace(160, 178, 10)])
    for i, a in enumerate(np.concatenate([[178] * 10] + [wob] * 5)):
        c.update({"knee": float(a)}, timestamp=i / 15)
    assert c.count == 0
