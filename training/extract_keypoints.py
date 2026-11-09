"""Étape 02 : passe MediaPipe sur chaque vidéo de data/raw/fitness_form/ et sauvegarde
points + features en Parquet dans data/processed/keypoints/ (un fichier par vidéo).

Chaque ligne = une frame : métadonnées (sujet, exercice, good/bad, vue), les 16 features
(coach.features.angles.FEATURE_NAMES) et les 33 landmarks bruts (lm{i}_x/_y/_v) pour pouvoir
recalculer d'autres features plus tard sans refaire la passe MediaPipe (lente).

Reprise : une vidéo déjà extraite est sautée. Relancer le script après une coupure suffit.

Usage :
    python training/extract_keypoints.py                  # tout, 4 processus
    python training/extract_keypoints.py --limit 3        # test sur 3 vidéos
    python training/extract_keypoints.py --workers 6 --exercise squat
    python training/extract_keypoints.py --merge-only     # recolle les fichiers -> all.parquet
"""
import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coach.features.angles import FEATURE_NAMES, compute_features
from coach.pose.estimator import PoseEstimator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fitness_form"
OUT = ROOT / "data" / "processed" / "keypoints"
NAME_RE = re.compile(r"^subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)\.mp4$")
EXERCISE_ALIASES = {"bicep": "bicep_curl"}  # un fichier du dataset est nommé « bicep »


def parse_name(path: Path) -> dict | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    subject, exercise, quality, view = m.groups()
    return {
        "video": path.name,
        "subject": int(subject),
        "exercise": EXERCISE_ALIASES.get(exercise, exercise),
        "quality": quality,
        "view": view,
    }


def extract_one(video_path: str) -> tuple[str, int, int]:
    """Traite une vidéo. Renvoie (nom, frames lues, frames avec pose détectée)."""
    path = Path(video_path)
    meta = parse_name(path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    est = PoseEstimator()
    rows, n, found = [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            ts_ms = int(n * 1000 / fps)
            lm = est.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms=ts_ms)
            row = {**meta, "frame": n, "t": n / fps, "width": w, "height": h, "detected": lm is not None}
            if lm is not None:
                found += 1
                row.update(compute_features(lm, aspect=w / h))
                for i, p in enumerate(lm):
                    row[f"lm{i}_x"], row[f"lm{i}_y"], row[f"lm{i}_v"] = p.x, p.y, p.visibility
            else:
                row.update({k: np.nan for k in FEATURE_NAMES})
            rows.append(row)
            n += 1
    finally:
        cap.release()
        est.close()
    df = pd.DataFrame(rows)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (path.stem + ".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT / (path.stem + ".parquet"))  # écriture atomique : pas de fichier à moitié écrit
    return path.name, n, found


def merge() -> Path:
    files = sorted(OUT.glob("*.parquet"))
    if not files:
        raise SystemExit("Aucun fichier à fusionner.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    out = ROOT / "data" / "processed" / "all.parquet"
    df.to_parquet(out, index=False)
    print(f"{out} : {len(df):,} frames, {df['video'].nunique()} vidéos, "
          f"{df['detected'].mean():.1%} frames avec pose détectée")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N vidéos (test)")
    ap.add_argument("--exercise", default=None, help="ex. squat, push_up, bicep_curl")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        merge()
        return

    videos = [p for p in sorted(RAW.glob("*.mp4")) if parse_name(p)]
    if args.exercise:
        videos = [p for p in videos if parse_name(p)["exercise"] == args.exercise]
    todo = [p for p in videos if not (OUT / (p.stem + ".parquet")).exists()]
    already_done = len(videos) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(videos)} vidéos, {already_done} déjà faites, "
          f"{len(todo)} à traiter avec {args.workers} processus", flush=True)
    if not todo:
        if not args.no_merge:
            merge()
        return

    t0, done, failed = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, str(p)): p for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                name, n, found = fut.result()
                done += 1
                eta = (time.time() - t0) / done * (len(todo) - done) / 60
                print(f"[{done}/{len(todo)}] {name}: {found}/{n} frames avec pose  (reste ~{eta:.0f} min)",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - une vidéo cassée ne doit pas arrêter les autres
                failed.append(p.name)
                print(f"ÉCHEC {p.name}: {e!r}", flush=True)

    print(f"Terminé en {(time.time() - t0) / 60:.1f} min. Échecs : {len(failed)}", flush=True)
    for f in failed:
        print("  -", f)
    if not args.no_merge and not failed:
        merge()


if __name__ == "__main__":
    main()
