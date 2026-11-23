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


def test_last_fix_follows_the_tip():
    fb = FeedbackEngine(encourage_after=1)
    lean = Fault("SQUAT_LEAN", "Chest up!", 1, "Brace your core and keep your chest proud.")
    assert fb.choose_tip([lean], now=0) == "Chest up!"
    assert fb.last_fix == "Brace your core and keep your chest proud."
    assert fb.choose_tip([], now=5) is not None  # encouragement : pas de correction
    assert fb.last_fix == ""


def test_spoken_cue_is_short_and_follows_the_tip():
    fb = FeedbackEngine(encourage_after=2)
    deeper = Fault("SQUAT_SHALLOW", "Go a bit lower", 2, "Sit down until thighs are parallel.", "Go deeper")
    assert fb.choose_tip([deeper], now=0) == "Go a bit lower" and fb.last_say == "Go deeper"
    assert fb.choose_tip([], now=5) is None              # 1 rep propre : rien
    assert fb.choose_tip([], now=6) is not None          # 2 reps propres : encouragement
    assert fb.last_say == "Great, keep going"


def test_fault_without_spoken_cue_reads_the_tip():
    fb = FeedbackEngine()
    fb.choose_tip([Fault("X", "Hands under your shoulders", 3)], now=0)
    assert fb.last_say == "Hands under your shoulders"
