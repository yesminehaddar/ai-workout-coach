"""Étape 6b : entraîne le LSTM qui reconnaît l'exercice (GPU si disponible).

Données : data/processed/all.parquet. Les features passent par JumpFilter + One Euro, comme
dans l'appli, puis sont découpées en fenêtres de 30 frames (pas de 5 en train, 15 en test).
Classes : squat, pushup, curl ; les 4 autres exercices du dataset -> idle.
Séparation par sujet : test = 5 sujets, validation (early stopping) = 3 sujets, train = le reste.
Adam, cross-entropy pondérée, early stopping. Suivi MLflow (mlruns/mlflow.db).

Usage : python training/train_classifier.py  -> models/exercise_lstm.pt (TorchScript)
"""
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from coach.config import load_config
from coach.features.angles import FEATURE_NAMES
from coach.features.smoothing import FeatureSmoother, JumpFilter
from coach.reasoning.classifier import CLASSES, ExerciseLSTM

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"squat": 0, "push_up": 1, "bicep_curl": 2}  # autre -> 3 (idle)
WIN = 30


def clip_arrays() -> list[tuple[np.ndarray, int, int, str, str]]:
    """Une entrée par clip : (features lissées (T, 16), label, sujet, vue, vidéo)."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "all.parquet",
                         columns=["video", "subject", "exercise", "view", "frame", "t", *FEATURE_NAMES])
    out = []
    for video, g in df.groupby("video"):
        g = g.sort_values("frame")
        jf, sm = JumpFilter(), FeatureSmoother()
        rows = []
        for t, f in zip(g["t"].to_numpy(), g[FEATURE_NAMES].to_dict("records")):
            rows.append([np.nan] * len(FEATURE_NAMES) if np.isnan(f["knee"])
                        else [sm(jf(f), t)[k] for k in FEATURE_NAMES])
        x = pd.DataFrame(rows).ffill().to_numpy(np.float32)
        out.append((x, LABELS.get(g["exercise"].iloc[0], 3), int(g["subject"].iloc[0]),
                    g["view"].iloc[0], video))
    return out


def windows(clips, stride: int):
    X, y, meta = [], [], []
    for x, label, subject, view, video in clips:
        for s in range(0, len(x) - WIN + 1, stride):
            X.append(x[s:s + WIN])
            y.append(label)
            meta.append((video, view))
    return np.stack(X), np.array(y), pd.DataFrame(meta, columns=["video", "view"])


@torch.no_grad()
def predict(model, X, device, bs=1024) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu().numpy()
                           for i in range(0, len(X), bs)])


def main():
    cfg = load_config("model")
    lc, tc = cfg["lstm"], cfg["training"]
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    clips = clip_arrays()
    subjects = sorted({c[2] for c in clips})
    rng = np.random.default_rng(0)
    perm = rng.permutation(subjects)
    test_s, val_s = set(perm[:5]), set(perm[5:8])
    split = {k: [c for c in clips if f(c[2])] for k, f in
             [("train", lambda s: s not in test_s | val_s), ("val", lambda s: s in val_s),
              ("test", lambda s: s in test_s)]}
    Xtr, ytr, _ = windows(split["train"], 5)
    Xva, yva, _ = windows(split["val"], 15)
    Xte, yte, mte = windows(split["test"], 15)
    print(f"Préparation {time.time() - t0:.0f} s : train {len(Xtr)}, val {len(Xva)}, test {len(Xte)} "
          f"fenêtres ; sujets test {sorted(int(s) for s in test_s)}")

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

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("exercise_classifier")
    best, best_state, patience = -1.0, None, 0
    with mlflow.start_run():
        mlflow.log_params({**lc, **tc, "window": WIN, "test_subjects": sorted(int(s) for s in test_s)})
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
        print(f"\nTest (5 sujets jamais vus) : {acc:.1%} des fenêtres, {clip_acc:.1%} des clips (vote)")
        print(pd.crosstab(mte.true, mte.pred, normalize="index").round(2).to_string())
        print(mte.groupby("view").ok.mean().round(3).to_string())

        out = ROOT / "models" / "exercise_lstm.pt"
        torch.jit.script(model.cpu().eval()).save(str(out))
        mlflow.log_artifact(str(out))
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
