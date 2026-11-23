"""Découpe chaque clip en répétitions et enregistre un résumé par rep.

Les features passent par la même chaîne que dans l'appli : JumpFilter -> One Euro -> RepCounter.
Chaque rep (comptée ou partielle) donne une ligne : métadonnées du clip, min/max de chaque
feature, amplitude (max - min), tempo. C'est l'entrée du FormAnalyzer (règles) et du
Random Forest (étape 6).

Sources : Mendeley (all.parquet, bonne / mauvaise forme par clip) et QEVD-FIT-300k (qevd.parquet,
défauts nommés -> quality + colonne faults, voir coach.datasets). Colonne source sur chaque ligne.

Usage :
    python ml/data/extract_rep_summaries.py                    # toutes les sources présentes
    python ml/data/extract_rep_summaries.py --sources mendeley
-> data/processed/reps.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coach.config import load_config
from coach.datasets import load_frames, parse_sources
from coach.features.angles import EXTRA_FEATURES, FEATURE_NAMES
from coach.features.smoothing import FeatureSmoother, JumpFilter
from coach.reasoning.rep_counter import RepCounter

ROOT = Path(__file__).resolve().parents[2]

# Nom dans le dataset -> nom dans rules.yaml
EXERCISES = {"squat": "squat", "push_up": "pushup", "bicep_curl": "curl"}
SUMMARY_FEATURES = [f for f in FEATURE_NAMES if f != "side"] + list(EXTRA_FEATURES)
META = ["video", "subject", "exercise", "quality", "view", "source", "group", "split", "faults"]


def rep_rows(g: pd.DataFrame, exercise: str, th: dict) -> list[dict]:
    jf, sm, rc = JumpFilter(), FeatureSmoother(), RepCounter(exercise, th)
    key = th["angle"]
    rows = []
    for t, f in zip(g["t"].to_numpy(), g[SUMMARY_FEATURES].to_dict("records")):
        if np.isnan(f[key]):
            continue
        rep = rc.update(sm(jf(f), t), timestamp=t)
        if rep is None:
            continue
        row = {"counted": rep["counted"], "tempo_s": rep["tempo_s"], "end_t": t}
        for k in SUMMARY_FEATURES:
            lo, hi = rep["features_min"].get(k, np.nan), rep["features_max"].get(k, np.nan)
            row[f"min_{k}"], row[f"max_{k}"], row[f"range_{k}"] = lo, hi, hi - lo
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="mendeley,qevd (défaut : toutes celles présentes)")
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    sources = parse_sources(args.sources, args.processed)
    rules = load_config("rules")
    df = load_frames(args.processed, sources, ["video", "exercise", "view", "frame", "t", *SUMMARY_FEATURES])
    df = df[df["exercise"].isin(EXERCISES) & df["quality"].notna()]  # QEVD inexploitable : écarté
    out_rows = []
    for video, g in df.groupby("video"):
        g = g.sort_values("frame")
        ex = EXERCISES[g["exercise"].iloc[0]]
        meta = {k: g[k].iloc[0] for k in META if k in g}
        meta["exercise"] = ex
        for i, r in enumerate(rep_rows(g, ex, rules["rep_counter"][ex])):
            out_rows.append({**meta, "rep": i, **r})
    res = pd.DataFrame(out_rows)
    out = args.processed / "reps.parquet"
    res.to_parquet(out, index=False)
    print(res.groupby(["source", "exercise", "view", "quality"])["counted"].agg(["size", "mean"])
          .rename(columns={"size": "reps", "mean": "counted_share"}).round(2).to_string())
    print(f"\n{len(res)} reps from {res['video'].nunique()} clips -> {out}")


if __name__ == "__main__":
    main()
