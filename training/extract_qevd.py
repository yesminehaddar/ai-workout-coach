"""Extraction des points et features pour les clips QEVD-FIT-300k (squats, pompes).

Entrée : data/raw/qevd_fit300k/clips/*.mp4 et clips_index.csv (écrits à partir des archives par
scripts/qevd_split_zip.py). QEVD n'a pas d'identifiant de personne : on garde sa séparation
train/test. Vue : toujours de face (webcam à la maison).

Usage : python training/extract_qevd.py [--workers 6]  -> data/processed/qevd.parquet
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from extract_keypoints import extract_video

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "qevd_fit300k"
OUT = ROOT / "data" / "processed" / "qevd_keypoints"
EXERCISE = {"squats": "squat", "pushups": "push_up"}  # mêmes noms que le dataset Mendeley


def job(row: dict) -> tuple[str, int, int]:
    meta = {"video": row["clip"], "exercise": EXERCISE[row["exercise"]], "labels": row["labels"],
            "split": row["split"], "view": "front", "source": "qevd"}
    return extract_video(RAW / "clips" / row["clip"], meta, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rows = pd.read_csv(RAW / "clips_index.csv").to_dict("records")
    todo = [r for r in rows if not (OUT / (Path(r["clip"]).stem + ".parquet")).exists()]
    print(f"{len(rows)} clips, {len(todo)} à traiter")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
            f.result()
    df = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    out = ROOT / "data" / "processed" / "qevd.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} frames, {df.video.nunique()} clips, pose détectée {df.detected.mean():.1%} -> {out}")


if __name__ == "__main__":
    main()
