"""Étape 6b : entraîne le LSTM qui reconnaît l'exercice (GPU si disponible).

Données : data/processed/all.parquet (Mendeley) et qevd.parquet (QEVD-FIT-300k, squats et pompes
de face), voir coach.datasets. Les features passent par JumpFilter + One Euro, comme dans l'appli,
puis sont découpées en fenêtres de 30 frames (pas de 5 en train, 15 en test).
Classes : squat, pushup, curl ; les 4 autres exercices Mendeley -> idle.
Séparation : Mendeley par sujet (test = 5 sujets, validation = 3), QEVD selon sa séparation
train / test (10 % du train en validation). Résultats test par source et par vue.
Adam, cross-entropy pondérée, early stopping. Suivi MLflow (data/mlruns/mlflow.db).

Usage :
    python ml/training/train_classifier.py                       # toutes les sources présentes
    python ml/training/train_classifier.py --sources mendeley    # refait le modèle d'origine
    python ml/training/train_classifier.py --out data/models/candidates/exercise_lstm.pt
Sortie par défaut : data/models/exercise_lstm.pt (TorchScript), le modèle utilisé par l'appli.
"""
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from tracking import start_experiment  # ml/training/tracking.py

from coach.config import load_config
from coach.datasets import load_frames, parse_sources
from coach.features.angles import FEATURE_NAMES
from coach.features.smoothing import FeatureSmoother, JumpFilter
from coach.reasoning.classifier import CLASSES, ExerciseLSTM

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"squat": 0, "push_up": 1, "bicep_curl": 2}  # autre -> 3 (idle)
WIN = 30


def clip_arrays(df: pd.DataFrame) -> list[tuple[np.ndarray, int, str, str, str, str]]:
    """Une entrée par clip : (features lissées (T, 16), label, split, vue, vidéo, source)."""
    out = []
    for video, g in df.groupby("video"):
        g = g.sort_values("frame")
        jf, sm = JumpFilter(), FeatureSmoother()
        rows = []
        for t, f in zip(g["t"].to_numpy(), g[FEATURE_NAMES].to_dict("records")):
            rows.append([np.nan] * len(FEATURE_NAMES) if np.isnan(f["knee"])
                        else [sm(jf(f), t)[k] for k in FEATURE_NAMES])
        x = pd.DataFrame(rows).ffill().to_numpy(np.float32)
        out.append((x, LABELS.get(g["exercise"].iloc[0], 3), g["split"].iloc[0],
                    g["view"].iloc[0], video, g["source"].iloc[0]))
    return out


def windows(clips, stride: int):
    X, y, meta = [], [], []
    for x, label, _split, view, video, source in clips:
        for s in range(0, len(x) - WIN + 1, stride):
            X.append(x[s:s + WIN])
            y.append(label)
            meta.append((video, view, source))
    return np.stack(X), np.array(y), pd.DataFrame(meta, columns=["video", "view", "source"])


@torch.no_grad()
def predict(model, X, device, bs=1024) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu().numpy()
                           for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="mendeley,qevd (défaut : toutes celles présentes)")
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "models" / "exercise_lstm.pt")
    args = ap.parse_args()
    sources = parse_sources(args.sources, args.processed)
    cfg = load_config("model")
    lc, tc = cfg["lstm"], cfg["training"]
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    frames = load_frames(args.processed, sources,
                         ["video", "subject", "exercise", "view", "frame", "t", *FEATURE_NAMES])
    test_groups = sorted(frames.loc[(frames.split == "test") & (frames.source == "mendeley"),
                                    "group"].unique())
    clips = clip_arrays(frames)
    del frames
    split = {k: [c for c in clips if c[2] == k] for k in ("train", "val", "test")}
    Xtr, ytr, _ = windows(split["train"], 5)
    Xva, yva, _ = windows(split["val"], 15)
    Xte, yte, mte = windows(split["test"], 15)
    print(f"Sources {sources}. Préparation {time.time() - t0:.0f} s : train {len(Xtr)}, val {len(Xva)}, "
          f"test {len(Xte)} fenêtres ; sujets Mendeley test {test_groups}")

    model = ExerciseLSTM(lc["input_features"], lc["hidden_size"], lc["num_layers"],
                         len(CLASSES), lc["dropout"])
    flat = Xtr.reshape(-1, Xtr.shape[2])
    model.mean.copy_(torch.from_numpy(np.nanmean(flat, 0)))
    model.std.copy_(torch.from_numpy(np.nanstd(flat, 0) + 1e-6))
    model.to(device)
    weights = torch.tensor(len(ytr) / (len(CLASSES) * np.bincount(ytr, minlength=len(CLASSES))),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)

    start_experiment("exercise_classifier")
    best, best_state, patience = -1.0, None, 0
    with mlflow.start_run():
        mlflow.log_params({**lc, **tc, "window": WIN, "sources": ",".join(sources),
                           "test_subjects": ",".join(test_groups)})
        for epoch in range(tc["max_epochs"]):
            model.train()
            order = torch.randperm(len(Xtr_t), device=device)
            for i in range(0, len(order), tc["batch_size"]):
                b = order[i:i + tc["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
                loss.backward()
                opt.step()
            val_acc = float((predict(model, Xva, device) == yva).mean())
            mlflow.log_metrics({"train_loss": loss.item(), "val_acc": val_acc}, step=epoch)
            if val_acc > best:
                best, patience = val_acc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  val {val_acc:.3f}")
            if patience >= tc["early_stopping_patience"]:
                break

        model.load_state_dict(best_state)
        pred = predict(model, Xte, device)
        mte["ok"] = pred == yte
        mte["true"], mte["pred"] = [CLASSES[i] for i in yte], [CLASSES[i] for i in pred]
        acc = float(mte.ok.mean())
        clip_vote = mte.groupby("video").agg(t=("true", "first"), p=("pred", lambda s: s.mode()[0]))
        clip_acc = float((clip_vote.t == clip_vote.p).mean())
        mlflow.log_metrics({"val_acc_best": best, "test_window_acc": acc, "test_clip_acc": clip_acc})
        print(f"\nTest (personnes / clips jamais vus) : {acc:.1%} des fenêtres, "
              f"{clip_acc:.1%} des clips (vote)")
        print(pd.crosstab(mte.true, mte.pred, normalize="index").round(2).to_string())
        print("\nPar source et par vue (part des fenêtres correctes, nombre) :")
        print(mte.groupby(["source", "view"]).ok.agg(["mean", "size"]).round(3).to_string())
        for (src, view), g in mte.groupby(["source", "view"]):
            mlflow.log_metric(f"test_window_acc_{src}_{view}", float(g.ok.mean()))

        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(model.cpu().eval()).save(str(out))
        mlflow.log_artifact(str(out))
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
