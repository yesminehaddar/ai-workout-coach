"""Test : les défauts de squat étiquetés dans QEVD sont-ils détectables avec nos features (vue de face) ?

Pour chaque défaut (peu profond, dos pas droit, genoux devant les orteils, pieds trop serrés) :
Random Forest « défaut vs pas ce défaut » sur un résumé du clip (percentiles 5/50/95 et amplitude
de chaque feature). Validation croisée 5 plis (QEVD n'a pas d'identifiant de personne : une même
personne peut être dans plusieurs plis, le score est donc un peu optimiste).
Courbe d'apprentissage : AUC avec 25 / 50 / 75 / 100 % des clips d'entraînement. Si elle monte
encore, plus de données aideraient.

Usage : python ml/evaluation/eval_qevd_squat.py   (après ml/data/extract_qevd.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from coach.features.angles import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
FEATS = [f for f in FEATURE_NAMES if f not in ("side", "visibility")]
FAULTS = {  # défaut -> étiquettes QEVD qui le signalent
    "shallow": {"shallow", "insufficient"},
    "back_not_straight": {"back not straight"},
    "knees_over_toes": {"knees over toes"},
    "narrow_stance": {"narrow"},
}
DEPTH = {"shallow", "insufficient", "90 degrees", "over 90 degrees"}


def clip_table() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "qevd.parquet")
    df = df[(df.exercise == "squat") & df.detected]
    rows = []
    for video, g in df.groupby("video"):
        tags = {x.split(" - ", 1)[1] for x in g.labels.iloc[0].split(" | ") if x.startswith("squats - ")}
        if not tags & DEPTH:  # « starting late », « hold » : pas un squat exploitable
            continue
        x = g.sort_values("frame")[FEATS].rolling(5, center=True, min_periods=1).median()
        r = {"video": video, "split": g.split.iloc[0], "tags": tags}
        for f in FEATS:
            p5, p50, p95 = np.nanpercentile(x[f], [5, 50, 95])
            r.update({f"p5_{f}": p5, f"p50_{f}": p50, f"p95_{f}": p95, f"range_{f}": p95 - p5})
        rows.append(r)
    return pd.DataFrame(rows)


def cv_auc(X, y, frac=1.0, seed=0) -> float:
    probs = np.zeros(len(y))
    rng = np.random.default_rng(seed)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        tr = rng.choice(tr, max(10, int(len(tr) * frac)), replace=False)
        m = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                   n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
        probs[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, probs)


def main():
    t = clip_table()
    cols = [c for c in t if c.startswith(("p5_", "p50_", "p95_", "range_"))]
    X = t[cols].fillna(0).to_numpy()
    print(f"{len(t)} clips de squat exploitables\n")
    rows = []
    for fault, tags in FAULTS.items():
        y = t.tags.map(lambda s, tags=tags: bool(s & tags)).to_numpy().astype(int)
        curve = [np.mean([cv_auc(X, y, f, s) for s in range(3)]) for f in (0.25, 0.5, 0.75, 1.0)]
        m = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1,
                                   random_state=0).fit(X, y)
        top = pd.Series(m.feature_importances_, cols).nlargest(3)
        rows.append({"fault": fault, "clips_with_fault": int(y.sum()),
                     "auc_25%": curve[0], "auc_50%": curve[1], "auc_75%": curve[2], "auc_100%": curve[3],
                     "top_features": ", ".join(top.index)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
