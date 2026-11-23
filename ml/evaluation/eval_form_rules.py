"""Vérifie les règles de posture (backend/coach/configs/rules.yaml) contre les étiquettes good/bad du dataset.

Le dataset dit seulement si un clip est « good » ou « bad », pas quel défaut. On mesure donc :
  - par règle : % de reps où elle se déclenche, sur les clips good (fausses alertes) et bad ;
  - par clip : part des reps avec au moins un défaut ; un clip est jugé « bad » au-delà de
    CLIP_THRESHOLD. Comparaison aussi par sujet : le clip bad a-t-il plus de défauts que le good
    du même sujet ? (élimine les différences entre personnes).

Entrée : data/processed/reps.parquet (python ml/data/extract_rep_summaries.py).
Usage : python ml/evaluation/eval_form_rules.py  -> data/keypoint_previews/form_rules_eval.csv
"""
from pathlib import Path

import pandas as pd

from coach.config import load_config
from coach.reasoning.form_analyzer import FormAnalyzer

ROOT = Path(__file__).resolve().parents[2]
VIEWS = {"squat": ["side"], "pushup": ["side", "diagonal"], "curl": ["side", "diagonal"]}
CLIP_THRESHOLD = 0.3


def main():
    fa = FormAnalyzer(load_config("rules"))
    reps = pd.read_parquet(ROOT / "data" / "processed" / "reps.parquet")
    reps = reps[[r.view in VIEWS[r.exercise] for r in reps[["exercise", "view"]].itertuples()]]

    rule_rows, rep_flags = [], []
    for ex, d in reps.groupby("exercise"):
        records = d.to_dict("records")
        fired = pd.DataFrame({r.code: [r.fires(m) for m in records] for r in fa.rules[ex]},
                             index=d.index)
        for code in fired:
            rule_rows.append({"exercise": ex, "rule": code,
                              "good_%": 100 * fired.loc[d.quality == "good", code].mean(),
                              "bad_%": 100 * fired.loc[d.quality == "bad", code].mean()})
        rep_flags.append(fired.any(axis=1).rename("faulty"))
    reps = reps.join(pd.concat(rep_flags))

    print("Par règle (% des reps où elle se déclenche) :")
    print(pd.DataFrame(rule_rows).round(0).to_string(index=False))

    clips = reps.groupby("video").agg(exercise=("exercise", "first"), subject=("subject", "first"),
                                      view=("view", "first"), quality=("quality", "first"),
                                      reps=("faulty", "size"), faulty_share=("faulty", "mean"))
    clips["predicted"] = (clips.faulty_share > CLIP_THRESHOLD).map({True: "bad", False: "good"})
    clips["correct"] = clips.predicted == clips.quality
    out = ROOT / "data" / "keypoint_previews" / "form_rules_eval.csv"
    clips.to_csv(out)

    paired = clips.pivot_table(index=["exercise", "subject", "view"], columns="quality",
                               values="faulty_share").dropna()
    paired["bad_worse"] = paired["bad"] > paired["good"]
    summary = clips.groupby("exercise").agg(
        clips=("correct", "size"), accuracy=("correct", "mean"),
        faulty_good=("faulty_share", lambda s: s[clips.loc[s.index, "quality"] == "good"].median()),
        faulty_bad=("faulty_share", lambda s: s[clips.loc[s.index, "quality"] == "bad"].median()))
    summary["bad_clip_worse_same_subject"] = paired.groupby("exercise")["bad_worse"].mean()
    print(f"\nPar clip (jugé « bad » si > {CLIP_THRESHOLD:.0%} des reps ont un défaut) :")
    print(summary.round(2).to_string())
    print(f"\nGlobal : {clips.correct.mean():.0%} des {len(clips)} clips bien classés -> {out}")


if __name__ == "__main__":
    main()
