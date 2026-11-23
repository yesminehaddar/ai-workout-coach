"""Étape 6a : Random Forest qui note chaque rep (probabilité « good » -> score 0-100).

Entrée : data/processed/reps.parquet (ml/data/extract_rep_summaries.py), vues fiables seulement :
Mendeley de profil (+ diagonale pompe / curl) et, si présent, QEVD squat et pompe de face (la pompe
de face, peu fiable dans Mendeley, est bien détectée dans QEVD : 75 % des clips en validation croisée). Un modèle par exercice. Étiquette d'une rep = étiquette de son clip.
Validation croisée par groupe (GroupKFold) : un sujet Mendeley n'est jamais à la fois en train et en
test ; QEVD n'a pas d'identifiant de personne, le groupe est le clip (score QEVD un peu optimiste).
Mesures : AUC par rep, et précision par clip (moyenne des probabilités > 0.5), à comparer aux
règles (ml/evaluation/eval_form_rules.py : 71 %).

Usage :
    python ml/training/train_form_model.py                      # toutes les sources présentes
    python ml/training/train_form_model.py --sources mendeley   # refait le modèle d'origine
    python ml/training/train_form_model.py --out data/models/candidates/form_model.joblib
Sortie par défaut : data/models/form_model.joblib (+ run MLflow dans data/mlruns/).
"""
import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tracking import start_experiment  # ml/training/tracking.py

from coach.config import load_config

ROOT = Path(__file__).resolve().parents[2]
VIEWS = {  # source -> exercice -> vues gardées
    "mendeley": {"squat": ["side"], "pushup": ["side", "diagonal"], "curl": ["side", "diagonal"]},
    "qevd": {"squat": ["front"], "pushup": ["front"]},
}
# Mesures qui décrivent le cadrage plus que le geste : exclues pour ne pas apprendre la caméra.
DROP = ("visibility", "shoulder_width")


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df if c.startswith(("min_", "max_", "range_"))] + ["tempo_s"]
    return [c for c in cols if not c.endswith(DROP)]


def make_model(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=cfg["n_estimators"], min_samples_leaf=5,
                                  class_weight="balanced", n_jobs=-1, random_state=0)


def evaluate(d: pd.DataFrame, cols: list[str], cfg: dict) -> dict:
    X, y = d[cols].fillna(0).to_numpy(), (d.quality == "good").to_numpy().astype(int)
    prob = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, d.group):
        prob[te] = make_model(cfg).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    clips = d.assign(p=prob).groupby("video").agg(p=("p", "mean"), q=("quality", "first"),
                                                  source=("source", "first"))
    clips["ok"] = (clips.p > 0.5) == (clips.q == "good")
    out = {"rep_auc": roc_auc_score(y, prob), "clip_accuracy": clips.ok.mean(),
           "reps": len(d), "clips": len(clips)}
    for src, g in clips.groupby("source"):
        out[f"clip_accuracy_{src}"] = g.ok.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="mendeley,qevd (défaut : toutes celles présentes dans reps.parquet)")
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "models" / "form_model.joblib")
    args = ap.parse_args()
    cfg = load_config("model")["random_forest"]
    reps = pd.read_parquet(args.processed / "reps.parquet")
    if "source" not in reps:  # reps.parquet d'avant QEVD
        reps = reps.assign(source="mendeley", group="m" + reps.subject.astype(int).astype(str))
    sources = args.sources.split(",") if args.sources else sorted(reps.source.unique())
    reps = reps[reps.source.isin(sources)]
    if "mendeley" not in sources:
        print("ATTENTION : sans Mendeley, seuls le squat et la pompe de face (QEVD) sont entraînés.")
    # Les clips test QEVD restent hors entraînement : ml/evaluation/eval_qevd.py compare les modèles dessus.
    held_out = (reps.source == "qevd") & (reps.split == "test")
    reps = reps[~held_out]
    if held_out.any():
        print(f"{held_out.sum()} reps des clips test QEVD gardées pour l'évaluation (non entraînées)")
    start_experiment("form_model")
    bundle = {}
    with mlflow.start_run():
        mlflow.log_params({**cfg, "split": "GroupKFold(5) by subject (QEVD: by clip)",
                           "sources": ",".join(sources)})
        for ex in ("squat", "pushup", "curl"):
            keep = [(reps.source == src) & (reps.exercise == ex) & reps.view.isin(VIEWS[src].get(ex, []))
                    for src in sources]
            d = reps[np.logical_or.reduce(keep)] if keep else reps.iloc[:0]
            if d.quality.nunique() < 2:
                print(f"{ex:7s} pas assez de données (bonnes et mauvaises reps) : ignoré")
                continue
            cols = feature_columns(d)
            m = evaluate(d, cols, cfg)
            mlflow.log_metrics({f"{ex}_{k}": float(v) for k, v in m.items()})
            per_src = "  ".join(f"{k.removeprefix('clip_accuracy_')} {v:.0%}" for k, v in m.items()
                                if k.startswith("clip_accuracy_"))
            print(f"{ex:7s} reps={m['reps']:4d} clips={m['clips']:3d}  "
                  f"AUC rep={m['rep_auc']:.2f}  clips corrects={m['clip_accuracy']:.0%}  ({per_src})")
            model = make_model(cfg).fit(d[cols].fillna(0).to_numpy(), d.quality == "good")
            top = pd.Series(model.feature_importances_, cols).nlargest(5)
            print("         plus utiles :", ", ".join(f"{k} {v:.2f}" for k, v in top.items()))
            bundle[ex] = {"model": model, "columns": cols}
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out)
        mlflow.log_artifact(str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
