"""Évaluation complète sur QEVD-FIT-300k (squats et pompes filmés de face, webcam à la maison).

1. Compteur de reps : comparé au comptage hors-ligne des creux de la courbe d'angle (même
   référence que eval_rep_counter.py ; QEVD n'a pas non plus de comptage manuel).
2. Règles de posture : chaque règle contre le défaut que QEVD nomme (rappel = part des reps de
   clips qui ont ce défaut où la règle se déclenche ; fausse alerte = part des reps de clips
   propres où elle se déclenche). Décision par clip : « bad » si plus de 30 % de reps fautives.
3. Score de forme (Random Forest) : modèles comparés sur les clips test QEVD (gardés hors de
   l'entraînement par ml/training/train_form_model.py) : AUC par rep, clips corrects.
4. Reconnaissance d'exercice (LSTM) : modèles comparés sur les clips test QEVD (jamais vus en
   entraînement par un modèle entraîné avec coach.datasets).

Usage :
    python ml/evaluation/eval_qevd.py
    python ml/evaluation/eval_qevd.py --form-models data/models/form_model.joblib data/models/candidates/form_model.joblib \\
        --lstm-models data/models/exercise_lstm.pt data/models/candidates/exercise_lstm.pt
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from coach.config import load_config
from coach.datasets import load_frames
from coach.features.angles import FEATURE_NAMES
from coach.features.smoothing import ANGLE_FEATURES
from coach.reasoning.classifier import CLASSES
from coach.reasoning.form_analyzer import FormAnalyzer, rep_view

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "training"))
from eval_rep_counter import run_clip
from train_classifier import clip_arrays, predict, windows

PROCESSED = ROOT / "data" / "processed"
EXERCISES = {"squat": "squat", "push_up": "pushup"}
# Règle -> défaut QEVD qu'elle doit détecter (None : QEVD n'a pas d'étiquette correspondante).
RULE_FAULT = {"SQUAT_SHALLOW": "shallow", "SQUAT_KNEES_FORWARD": "knees_over_toes",
              "SQUAT_LEAN": "back_not_straight", "SQUAT_FAST": None,
              "SQUAT_NARROW": "narrow_stance", "SQUAT_KNEES_IN": None,
              "PUSHUP_SAG": "hips_sag", "PUSHUP_SHALLOW": "shallow", "PUSHUP_HANDS_FORWARD": None}
CLIP_THRESHOLD = 0.3


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def rep_counter(frames: pd.DataFrame) -> pd.DataFrame:
    rules = load_config("rules")["rep_counter"]
    rows = []
    for (video, ex), g in frames[frames.quality.notna()].groupby(["video", "exercise"]):
        r = run_clip(g.sort_values("frame"), EXERCISES[ex], rules[EXERCISES[ex]])
        faults = g.faults.iloc[0]
        rows.append({"video": video, "exercise": EXERCISES[ex],
                     "group": "shallow" if "shallow" in faults else g.quality.iloc[0], **r})
    res = pd.DataFrame(rows)
    res["diff"] = res.counted - res.reference
    res["exact"], res["within_1"] = res["diff"] == 0, res["diff"].abs() <= 1
    res["zero_counted"] = (res.counted == 0) & (res.reference > 0)
    return res


def form_rules(reps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fa = FormAnalyzer(load_config("rules"))
    rule_rows, clip_rows = [], []
    for ex, d in reps.groupby("exercise"):
        metrics = [rep_metrics_row(m) for m in d.to_dict("records")]
        views = [rep_view(m) for m in metrics]
        codes = dict.fromkeys(r.code for r in fa.rules[ex])  # une règle par vue peut porter le même code
        fired = pd.DataFrame({c: [any(r.applies(m, v) for r in fa.rules[ex] if r.code == c)
                                  for m, v in zip(metrics, views)] for c in codes}, index=d.index)
        clean = d.quality == "good"
        for code in fired:
            fault = RULE_FAULT.get(code)
            has = d.faults.str.split(",").map(lambda f, fault=fault: fault in f) if fault else None
            rule_rows.append({
                "exercise": ex, "rule": code, "qevd_fault": fault or "-",
                "recall_%": 100 * fired.loc[has, code].mean() if fault and has.any() else np.nan,
                "false_alarm_%": 100 * fired.loc[clean, code].mean(),
                "reps_with_fault": int(has.sum()) if fault else 0})
        d = d.assign(faulty=fired.any(axis=1))
        clips = d.groupby("video").agg(q=("quality", "first"), share=("faulty", "mean"))
        clips["ok"] = (clips.share > CLIP_THRESHOLD) == (clips.q == "bad")
        clip_rows.append({"exercise": ex, "clips": len(clips), "clips_correct_%": 100 * clips.ok.mean(),
                          "good_clips_judged_good_%": 100 * clips.loc[clips.q == "good", "ok"].mean(),
                          "bad_clips_judged_bad_%": 100 * clips.loc[clips.q == "bad", "ok"].mean()})
    return pd.DataFrame(rule_rows), pd.DataFrame(clip_rows)


def rep_metrics_row(row: dict) -> dict:
    """Ligne de reps.parquet (min_X, max_X, range_X, tempo_s) -> mesures des règles."""
    return {k: v for k, v in row.items() if k.startswith(("min_", "max_", "range_")) or k == "tempo_s"}


def form_models(reps: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        bundle = joblib.load(path)
        for ex, d in reps.groupby("exercise"):
            entry = bundle.get(ex)
            if entry is None:
                rows.append({"model": path.name if path.parent.name == "models" else f"{path.parent.name}/{path.name}",
                             "exercise": ex, "note": "pas de modèle pour cet exercice"})
                continue
            X = d.reindex(columns=entry["columns"]).fillna(0).to_numpy()
            p = entry["model"].predict_proba(X)[:, 1]
            y = (d.quality == "good").to_numpy()
            clips = d.assign(p=p).groupby("video").agg(p=("p", "mean"), q=("quality", "first"))
            ok = (clips.p > 0.5) == (clips.q == "good")
            rows.append({"model": f"{path.parent.name}/{path.name}", "exercise": ex,
                         "rep_auc": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
                         "clips_correct_%": 100 * ok.mean(),
                         "good_clips_ok_%": 100 * ok[clips.q == "good"].mean(),
                         "bad_clips_ok_%": 100 * ok[clips.q == "bad"].mean(),
                         "clips": len(clips), "note": ""})
    return pd.DataFrame(rows)


def lstm_models(frames: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    test = frames[frames.split == "test"]
    clips = clip_arrays(test)
    X, y, meta = windows(clips, 15)
    rows = []
    for path in paths:
        if not path.exists():
            continue
        model = torch.jit.load(str(path), map_location="cpu")
        pred = predict(model, X, "cpu")
        meta = meta.assign(true=[CLASSES[i] for i in y], pred=[CLASSES[i] for i in pred])
        vote = meta.groupby("video").agg(t=("true", "first"), p=("pred", lambda s: s.mode()[0]))
        for ex in ("squat", "pushup"):
            m, v = meta[meta.true == ex], vote[vote.t == ex]
            rows.append({"model": f"{path.parent.name}/{path.name}", "exercise": ex,
                         "windows_correct_%": 100 * (m.true == m.pred).mean(),
                         "clips_correct_%": 100 * (v.t == v.p).mean(), "test_clips": len(v),
                         "most_common_mistake": (m.pred[m.pred != ex].mode().iloc[0]
                                                 if (m.pred != ex).any() else "-")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-models", nargs="*", type=Path,
                    default=[ROOT / "data/models/form_model.joblib", ROOT / "data/models/candidates/form_model.joblib"])
    ap.add_argument("--lstm-models", nargs="*", type=Path,
                    default=[ROOT / "data/models/exercise_lstm.pt", ROOT / "data/models/candidates/exercise_lstm.pt"])
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    frames = load_frames(PROCESSED, ["qevd"], ["video", "exercise", "view", "frame", "t", "detected",
                                               *dict.fromkeys([*ANGLE_FEATURES, *FEATURE_NAMES])])
    print(f"QEVD : {frames.video.nunique()} clips, {len(frames):,} frames "
          f"({frames.groupby('video').quality.first().isna().sum()} clips inexploitables écartés de 1-3)")

    section("1. COMPTEUR DE REPS (vue de face) vs comptage hors-ligne")
    rc = rep_counter(frames)
    print(rc.groupby(["exercise", "group"]).agg(
        clips=("video", "size"), exact_pct=("exact", "mean"), within_1_pct=("within_1", "mean"),
        zero_counted_pct=("zero_counted", "mean"), counted_median=("counted", "median"),
        reference_median=("reference", "median")).assign(
        exact_pct=lambda d: (100 * d.exact_pct).round(0), within_1_pct=lambda d: (100 * d.within_1_pct).round(0),
        zero_counted_pct=lambda d: (100 * d.zero_counted_pct).round(0)).to_string())
    for ex, d in rc.groupby("exercise"):
        print(f"  {ex:7s} global : {d.exact.mean():.0%} exacts, {d.within_1.mean():.0%} à ±1 rep ({len(d)} clips)")

    reps = pd.read_parquet(PROCESSED / "reps.parquet")
    reps = reps[(reps.source == "qevd") & reps.counted]

    section("2. RÈGLES DE POSTURE (corrections) vs défauts nommés par QEVD")
    rules, clips = form_rules(reps)
    print(rules.round(1).to_string(index=False))
    print()
    print(clips.round(0).to_string(index=False))

    section("3. SCORE DE FORME (Random Forest) sur les clips test QEVD (jamais entraînés)")
    fm = form_models(reps[reps.split == "test"], args.form_models)
    print(fm.round(2).to_string(index=False) if len(fm) else "aucun modèle trouvé")

    section("4. RECONNAISSANCE D'EXERCICE (LSTM) sur les clips test QEVD")
    lm = lstm_models(frames, args.lstm_models)
    print(lm.round(1).to_string(index=False) if len(lm) else "aucun modèle trouvé")


if __name__ == "__main__":
    main()
