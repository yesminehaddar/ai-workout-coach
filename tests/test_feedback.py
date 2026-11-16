"""Moteur de conseils : priorité, cooldown, encouragement."""
from coach.feedback.engine import FeedbackEngine
from coach.reasoning.form_analyzer import Fault

LEAN = Fault("SQUAT_LEAN", "Chest up!", 1)
SHALLOW = Fault("SQUAT_SHALLOW", "Go a bit lower", 2)


def test_highest_priority_first():
    fb = FeedbackEngine()
    assert fb.choose_tip([SHALLOW, LEAN], now=0) == "Chest up!"


def test_cooldown_then_next_fault():
    fb = FeedbackEngine(cooldown_s=3)
    fb.choose_tip([LEAN, SHALLOW], now=0)
    assert fb.choose_tip([LEAN, SHALLOW], now=1) == "Go a bit lower"  # LEAN en cooldown
    assert fb.choose_tip([LEAN, SHALLOW], now=2) is None              # les deux en cooldown
    assert fb.choose_tip([LEAN], now=3.5) == "Chest up!"


def test_encourage_after_clean_reps():
    fb = FeedbackEngine(encourage_after=3)
    assert fb.choose_tip([], 0) is None and fb.choose_tip([], 1) is None
    assert fb.choose_tip([], 2) is not None
    assert len(fb.history) == 1


def test_low_score_without_rule_gets_general_tip():
    fb = FeedbackEngine(encourage_after=1)
    assert fb.choose_tip([], 0, score=20) == "Slow down and control the full movement"
    assert fb.choose_tip([], 5, score=90) is not None  # bon score : encouragement
