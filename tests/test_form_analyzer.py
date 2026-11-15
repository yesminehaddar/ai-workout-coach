"""Tests des règles de posture : parsing, déclenchement, tri par priorité, score par défaut."""
import numpy as np
import pytest

from coach.config import load_config
from coach.reasoning.form_analyzer import FormAnalyzer, Rule, rep_metrics
from coach.reasoning.rep_counter import RepCounter

RULES = {"squat": [
    {"code": "SQUAT_LEAN", "check": "max_torso_lean > 55", "tip": "Chest up!", "priority": 1},
    {"code": "SQUAT_SHALLOW", "check": "min_knee > 85", "tip": "Go a bit lower", "priority": 2},
    {"code": "SQUAT_FAST", "check": "tempo_s < 1.0", "tip": "Slow down", "priority": 3},
]}


def summary(knee=(70, 175), lean=(5, 30), tempo=2.0):
    return {"exercise": "squat", "tempo_s": tempo,
            "features_min": {"knee": knee[0], "torso_lean": lean[0]},
            "features_max": {"knee": knee[1], "torso_lean": lean[1]}}


def test_rep_metrics_flattens_min_max_range():
    m = rep_metrics(summary(knee=(70, 175)))
    assert m["min_knee"] == 70 and m["max_knee"] == 175 and m["range_knee"] == 105
    assert m["tempo_s"] == 2.0


def test_clean_rep_has_no_fault_and_full_score():
    fa = FormAnalyzer({"form_rules": RULES})
    s = summary()
    assert fa.check_rules("squat", s) == []
    assert fa.form_score(s) == 100


def test_faults_sorted_by_priority():
    fa = FormAnalyzer(RULES)
    s = summary(knee=(95, 175), lean=(5, 60), tempo=0.8)
    assert [f.code for f in fa.check_rules("squat", s)] == ["SQUAT_LEAN", "SQUAT_SHALLOW", "SQUAT_FAST"]
    assert fa.form_score(s) == 100 - 30 - 20 - 10


def test_missing_metric_does_not_fire():
    fa = FormAnalyzer(RULES)
    s = {"exercise": "squat", "tempo_s": 2.0, "features_min": {}, "features_max": {}}
    assert fa.check_rules("squat", s) == []


def test_unknown_exercise_has_no_rules():
    assert FormAnalyzer(RULES).check_rules("plank", summary()) == []


def test_invalid_check_raises():
    with pytest.raises(ValueError):
        Rule.parse({"code": "X", "check": "min_knee ~ 3", "tip": "", "priority": 1})


def test_project_rules_parse():
    cfg = load_config("rules")
    fa = FormAnalyzer(cfg)
    assert {"squat", "pushup", "curl"} <= fa.rules.keys()


def test_works_on_rep_counter_output():
    """Un rep réel du RepCounter passe directement dans le FormAnalyzer."""
    cfg = load_config("rules")
    rc = RepCounter("squat", cfg["rep_counter"]["squat"])
    fa = FormAnalyzer(cfg)
    down = np.linspace(175, 95, 30)  # descend sous 100 mais pas sous 85 : un peu haut
    reps = []
    for i, k in enumerate(np.concatenate([[175] * 5, down, down[::-1]])):
        e = rc.update({"knee": float(k), "torso_lean": 20.0}, timestamp=i / 30)
        if e:
            reps.append(e)
    assert len(reps) == 1 and reps[0]["counted"]
    assert [f.code for f in fa.check_rules("squat", reps[0])] == ["SQUAT_SHALLOW"]
