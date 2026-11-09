"""Aperçu visuel de la détection : vidéos annotées (squelette) + rapport de qualité.

Lit data/processed/keypoints/*.parquet (pas de nouvelle passe MediaPipe) et dessine :
  - le squelette (bleu = côté gauche, orange = côté droit, vert = nez),
  - les angles genou / coude,
  - un cadre ROUGE « GLITCH » quand un angle saute de plus de JUMP_DEG degrés d'une frame à l'autre,
  - « NO POSE » quand rien n'est détecté.

Sorties dans data/keypoint_previews/ :
  quality_report.csv   une ligne par vidéo (les 1042) avec les indicateurs de qualité
  overview.jpg         une image par clip de l'échantillon
  clips/*.mp4          vidéos annotées de l'échantillon

Usage :
    python scripts/render_keypoints.py                      # échantillon par défaut
    python scripts/render_keypoints.py --video subject_003_squat_good_diagonal.mp4
    python scripts/render_keypoints.py --report-only
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coach.pose.skeleton import draw_skeleton, put_text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fitness_form"
KP = ROOT / "data" / "processed" / "keypoints"
OUT = ROOT / "data" / "keypoint_previews"

JUMP_DEG = 40.0          # saut d'angle entre deux frames considéré comme une erreur de détection
FLAG_JUMP_PCT = 3.0      # une vidéo est « à vérifier » au-dessus de ce % de frames en saut
HEIGHT = 720             # hauteur des vidéos de sortie

SAMPLE_EXERCISES = ["squat", "push_up", "bicep_curl"]
SAMPLE_SUBJECTS = [1, 2]


def add_jumps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("frame").copy()
    ok = df["detected"]
    for col in ("knee", "elbow"):
        # écart avec la dernière frame détectée
        d = df[col].where(ok).ffill().diff().abs()
        df[f"jump_{col}"] = (d > JUMP_DEG) & ok
    df["glitch"] = df["jump_knee"] | df["jump_elbow"]
    return df


def quality_report() -> pd.DataFrame:
    rows = []
    for f in sorted(KP.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["video", "subject", "exercise", "quality", "view",
                                         "frame", "width", "height", "detected", "knee",
                                         "elbow", "visibility"])
        if df.empty:
            continue
        df = add_jumps(df)
        det = df[df["detected"]]
        rows.append({
            "video": df["video"].iloc[0], "subject": df["subject"].iloc[0],
            "exercise": df["exercise"].iloc[0], "quality": df["quality"].iloc[0],
            "view": df["view"].iloc[0], "width": df["width"].iloc[0], "height": df["height"].iloc[0],
            "frames": len(df), "detected_pct": round(100 * df["detected"].mean(), 1),
            "mean_visibility": round(float(det["visibility"].mean()), 3) if len(det) else 0.0,
            "knee_jump_pct": round(100 * df["jump_knee"].mean(), 2),
            "elbow_jump_pct": round(100 * df["jump_elbow"].mean(), 2),
            "glitch_pct": round(100 * df["glitch"].mean(), 2),
        })
    rep = pd.DataFrame(rows)
    rep["check_needed"] = (rep["glitch_pct"] > FLAG_JUMP_PCT) | (rep["detected_pct"] < 90)
    return rep


def draw(frame, row, label, glitch):
    h, w = frame.shape[:2]
    if row is not None and row["detected"]:
        draw_skeleton(frame, {i: (row[f"lm{i}_x"], row[f"lm{i}_y"]) for i in range(33)})
        info = f"knee {row['knee']:.0f}  elbow {row['elbow']:.0f}"
    else:
        info = "NO POSE"
    if glitch:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), max(6, h // 80))
        info += "   GLITCH"
    scale = max(0.4, 0.84 * h / 1000)
    x = int(0.02 * w)
    put_text(frame, label, (x, int(0.045 * h)), scale)
    put_text(frame, info, (x, int(0.085 * h)), scale, (80, 80, 255) if glitch else (0, 255, 255))
    return frame


def render_clip(video: str, out_dir: Path) -> tuple[np.ndarray, int]:
    """Écrit la vidéo annotée. Renvoie (une image représentative, nb de frames glitch)."""
    df = add_jumps(pd.read_parquet(KP / (Path(video).stem + ".parquet"))).reset_index(drop=True)
    cap = cv2.VideoCapture(str(RAW / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    size = (int(round(w0 * HEIGHT / h0 / 2) * 2), HEIGHT)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    label = video.replace("subject_", "s").replace(".mp4", "")
    mid, snap, n = len(df) // 2, None, 0
    while True:
        ok, frame = cap.read()
        if not ok or n >= len(df):
            break
        frame = cv2.resize(frame, size)
        out = draw(frame, df.iloc[n], label, bool(df.iloc[n]["glitch"]))
        writer.write(out)
        if n == mid:
            snap = out.copy()
        n += 1
    cap.release()
    writer.release()
    return snap, int(df["glitch"].sum())


def choose_sample(rep: pd.DataFrame) -> list[str]:
    s = rep[rep["exercise"].isin(SAMPLE_EXERCISES) & rep["subject"].isin(SAMPLE_SUBJECTS)]
    vids = list(s["video"])
    # + le clip le plus « glitché » de chaque exercice, pour voir le pire cas
    for ex in SAMPLE_EXERCISES:
        worst = rep[rep["exercise"] == ex].sort_values("glitch_pct", ascending=False).iloc[0]["video"]
        if worst not in vids:
            vids.append(worst)
    return sorted(vids)


def overview(snaps: list[np.ndarray]) -> np.ndarray:
    h = 360
    tiles = [cv2.resize(s, (int(s.shape[1] * h / s.shape[0]), h)) for s in snaps]
    rows, line, width = [], [], 0
    for t in tiles:
        if width + t.shape[1] > 2400 and line:
            rows.append(line); line, width = [], 0
        line.append(t); width += t.shape[1]
    rows.append(line)
    canvas_w = max(sum(t.shape[1] for t in r) for r in rows)
    imgs = []
    for r in rows:
        row_img = np.hstack(r)
        pad = np.zeros((h, canvas_w - row_img.shape[1], 3), np.uint8)
        imgs.append(np.hstack([row_img, pad]))
    return np.vstack(imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", help="rendre seulement ce(s) clip(s)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rep = quality_report()
    rep.to_csv(OUT / "quality_report.csv", index=False)
    print(f"quality_report.csv : {len(rep)} vidéos, {int(rep['check_needed'].sum())} à vérifier", flush=True)
    if args.report_only:
        return

    vids = args.video or choose_sample(rep)
    snaps = []
    for i, v in enumerate(vids, 1):
        snap, g = render_clip(v, OUT / "clips")
        snaps.append(snap)
        print(f"[{i}/{len(vids)}] {v}  glitches : {g}", flush=True)
    if not args.video:
        cv2.imwrite(str(OUT / "overview.jpg"), overview(snaps), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Terminé -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
