"""Évalue le compteur de répétitions sur le dataset extrait (data/processed/all.parquet).

Le dataset n'a pas de comptage manuel. On compare donc à une référence indépendante :
les creux de la courbe d'angle détectés hors-ligne (scipy find_peaks, qui voit toute la vidéo
d'un coup). Un grand écart = clip à regarder (vidéos annotées : scripts/render_keypoints.py).

Le compteur reçoit les features comme dans l'appli : JumpFilter -> One Euro -> RepCounter.

Usage : python scripts/eval_rep_counter.py   -> data/keypoint_previews/rep_count_eval.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from coach.features.smoothing import ANGLE_FEATURES, FeatureSmoother, JumpFilter
from coach.reasoning.rep_counter import RepCounter

ROOT = Path(__file__).resolve().parents[1]

# Nom dans le dataset -> nom dans rules.yaml, et vues fiables (cf. quality_report.csv)
EXERCISES = {"squat": ("squat", ["side", "diagonal"]),
             "push_up": ("pushup", ["side", "diagonal"]),
             "bicep_curl": ("curl", ["side", "diagonal"])}


def reference_count(x: np.ndarray) -> int:
    x = pd.Series(x).rolling(5, center=True, min_periods=1).median().to_numpy()
    p5, p95 = np.percentile(x, [5, 95])
    peaks, _ = find_peaks(-x, prominence=max((p95 - p5) * 0.5, 20), distance=8)
    return len(peaks)


def run_clip(g: pd.DataFrame, exercise: str, th: dict) -> dict:
    jf, sm = JumpFilter(), FeatureSmoother()
    rc = RepCounter(exercise, th)
    key = th["angle"]
    cols = [c for c in ANGLE_FEATURES if c in g]
    for t, row in zip(g["t"].to_numpy(), g[cols].to_dict("records")):
        if np.isnan(row[key]):  # pas de pose
            continue
        rc.update(sm(jf(row), t), timestamp=t)
    return {"counted": rc.count, "partial": rc.partial_count,
            "reference": reference_count(g[key].dropna().to_numpy())}


def main():
    rules = yaml.safe_load((ROOT / "configs" / "rules.yaml").read_text(encoding="utf-8"))
    cols = ["video", "subject", "exercise", "quality", "view", "frame", "t", "detected",
            *ANGLE_FEATURES]
    df = pd.read_parquet(ROOT / "data" / "processed" / "all.parquet", columns=cols)
    rows = []
    for ds_name, (ex, views) in EXERCISES.items():
        d = df[(df["exercise"] == ds_name) & df["view"].isin(views)]
        th = rules["rep_counter"][ex]
        for video, g in d.groupby("video"):
            g = g.sort_values("frame")
            r = run_clip(g, ex, th)
            rows.append({"video": video, "exercise": ex, "quality": g["quality"].iloc[0],
                         "view": g["view"].iloc[0], **r})
    res = pd.DataFrame(rows)
    res["diff"] = res["counted"] - res["reference"]
    out = ROOT / "data" / "keypoint_previews" / "rep_count_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    res["exact"] = res["diff"] == 0
    res["within1"] = res["diff"].abs() <= 1
    summary = res.groupby(["exercise", "view"]).agg(
        clips=("video", "size"), exact=("exact", "mean"), within_1=("within1", "mean"),
        mean_abs_diff=("diff", lambda s: s.abs().mean()), counted=("counted", "median"),
        reference=("reference", "median"), partial=("partial", "median"))
    print(summary.round(2).to_string())
    print(f"\nGlobal : {res['exact'].mean():.0%} exacts, {res['within1'].mean():.0%} à ±1 rep "
          f"({len(res)} clips) -> {out}")


if __name__ == "__main__":
    main()
