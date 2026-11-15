"""Étape 6a : Random Forest qui note chaque rep (probabilité « good » -> score 0-100).

Entrée : data/processed/reps.parquet (training/extract_rep_summaries.py), vues fiables seulement.
Un modèle par exercice. Étiquette d'une rep = étiquette de son clip (le dataset n'en a pas d'autre).
Validation croisée par sujet (GroupKFold) : un sujet n'est jamais à la fois en train et en test.
Mesures : AUC par rep, et précision par clip (moyenne des probabilités > 0.5), à comparer aux
règles (scripts/eval_form_rules.py : 71 %).

Usage : python training/train_form_model.py  -> models/form_model.joblib (+ run MLflow dans mlruns/)
"""
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from coach.config import load_config

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {"squat": ["side"], "pushup": ["side", "diagonal"], "curl": ["side", "diagonal"]}
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
    for tr, te in GroupKFold(n_splits=5).split(X, y, d.subject):
        prob[te] = make_model(cfg).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    clips = d.assign(p=prob).groupby("video").agg(p=("p", "mean"), q=("quality", "first"))
    return {"rep_auc": roc_auc_score(y, prob),
            "clip_accuracy": ((clips.p > 0.5) == (clips.q == "good")).mean(),
            "reps": len(d), "clips": len(clips)}


def main():
    cfg = load_config("model")["random_forest"]
    reps = pd.read_parquet(ROOT / "data" / "processed" / "reps.parquet")
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("form_model")
    bundle = {}
    with mlflow.start_run():
        mlflow.log_params({**cfg, "split": "GroupKFold(5) by subject"})
        for ex, views in VIEWS.items():
            d = reps[(reps.exercise == ex) & reps.view.isin(views)]
            cols = feature_columns(d)
            m = evaluate(d, cols, cfg)
            mlflow.log_metrics({f"{ex}_{k}": float(v) for k, v in m.items()})
            print(f"{ex:7s} reps={m['reps']:4d} clips={m['clips']:3d}  "
                  f"AUC rep={m['rep_auc']:.2f}  clips corrects={m['clip_accuracy']:.0%}")
            model = make_model(cfg).fit(d[cols].fillna(0).to_numpy(), d.quality == "good")
            top = pd.Series(model.feature_importances_, cols).nlargest(5)
            print("         plus utiles :", ", ".join(f"{k} {v:.2f}" for k, v in top.items()))
            bundle[ex] = {"model": model, "columns": cols}
        out = ROOT / "models" / "form_model.joblib"
        joblib.dump(bundle, out)
        mlflow.log_artifact(str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
