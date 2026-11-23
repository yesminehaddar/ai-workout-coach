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


def test_fault_carries_fix_text():
    rules = {"squat": [{"code": "SQUAT_SHALLOW", "check": "min_knee > 85", "tip": "Go a bit lower",
                        "priority": 2, "fix": "Sit down until thighs are parallel."}]}
    faults = FormAnalyzer(rules).check_rules("squat", summary(knee=(95, 175)))
    assert faults[0].fix == "Sit down until thighs are parallel."


def test_every_project_rule_explains_how_to_correct():
    fa = FormAnalyzer(load_config("rules"))
    missing = [r.code for rules in fa.rules.values() for r in rules if len(r.fix) < 40]
    assert missing == []


def test_verdict_good_or_bad():
    from coach.reasoning.form_analyzer import Fault, verdict
    assert verdict([], 100) == "good"
    assert verdict([], None) == "good"
    assert verdict([], 30) == "bad"  # aucune règle, mais score du modèle bas
    assert verdict([Fault("SQUAT_SHALLOW", "Go a bit lower", 2)], 90) == "bad"


def test_every_project_rule_has_a_short_spoken_cue():
    fa = FormAnalyzer(load_config("rules"))
    too_long = [(r.code, r.say) for rules in fa.rules.values() for r in rules
                if not r.say or len(r.say.split()) > 3]
    assert too_long == []


def test_camera_view_from_shoulder_width():
    from coach.reasoning.form_analyzer import rep_view
    assert rep_view({"min_shoulder_width": 0.7, "max_shoulder_width": 0.9}) == "front"
    assert rep_view({"min_shoulder_width": 0.05, "max_shoulder_width": 0.3}) == "side"
    assert rep_view({}) == "side"  # inconnue : la vue conseillée


def test_rules_only_apply_in_their_view():
    rules = {"squat": [
        {"code": "SQUAT_SHALLOW", "check": "min_knee > 85", "view": "side", "tip": "Go a bit lower", "priority": 2},
        {"code": "SQUAT_KNEES_IN", "check": "min_knee_spread < 0.8", "view": "front", "tip": "Knees out",
         "priority": 1}]}
    fa = FormAnalyzer(rules)
    side = {"exercise": "squat", "tempo_s": 2.0,
            "features_min": {"knee": 95, "knee_spread": 0.6, "shoulder_width": 0.1},
            "features_max": {"knee": 175, "knee_spread": 1.0, "shoulder_width": 0.2}}
    front = {**side, "features_min": {**side["features_min"], "shoulder_width": 0.75},
             "features_max": {**side["features_max"], "shoulder_width": 0.85}}
    assert [f.code for f in fa.check_rules("squat", side)] == ["SQUAT_SHALLOW"]
    assert [f.code for f in fa.check_rules("squat", front)] == ["SQUAT_KNEES_IN"]


def test_invalid_view_raises():
    with pytest.raises(ValueError):
        Rule.parse({"code": "X", "check": "min_knee > 3", "view": "top", "tip": "", "priority": 1})


def test_project_squat_rules_cover_back_knees_and_stance_from_the_front():
    fa = FormAnalyzer(load_config("rules"))
    front = {r.code for r in fa.rules["squat"] if r.view in ("front", "any")}
    assert {"SQUAT_LEAN", "SQUAT_KNEES_IN", "SQUAT_NARROW", "SQUAT_SHALLOW"} <= front
