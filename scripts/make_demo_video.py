"""Vidéo de démo de secours : les 6 clips de démo passés dans le coach, avec un carton titre
avant chaque clip et le bilan à la fin. À montrer si l'application ou la webcam font défaut.

Usage : python scripts/make_demo_video.py  -> demo/coach_demo.mp4 (1280x720, 30 fps, H.264)
Le H.264 passe par imageio-ffmpeg (pip install imageio-ffmpeg) ; sans lui, le fichier reste en
MPEG-4 brut, lisible mais ~20 fois plus gros.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from coach.config import load_config
from coach.pipeline import CoachPipeline
from coach.pose.skeleton import put_text
from coach.ui import CLIPS, DEMO_CLIPS

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 30
BG = (23, 17, 15)


def fit(img: np.ndarray) -> np.ndarray:
    """Place l'image (portrait ou paysage) au centre d'un cadre 1280x720."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.full((H, W, 3), BG, np.uint8)
    y, x = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def card(lines: list[tuple[str, float, tuple]], seconds: float = 2.5) -> list[np.ndarray]:
    img = np.full((H, W, 3), BG, np.uint8)
    y = H // 2 - 40 * (len(lines) - 1)
    for text, scale, color in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        put_text(img, text, ((W - tw) // 2, y), scale, color)
        y += int(80 * scale)
    return [img] * int(seconds * FPS)


def open_writer(path: Path) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):  # H.264 si OpenCV le permet, sinon MPEG-4
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
        if wr.isOpened():
            return wr
    raise RuntimeError("aucun codec vidéo disponible")


def main():
    app, rules = load_config("app"), load_config("rules")
    out = ROOT / "demo" / "coach_demo.mp4"
    raw = out.with_name("_raw.mp4")
    out.parent.mkdir(exist_ok=True)
    wr = open_writer(raw)

    def write(frames):
        for f in frames:
            wr.write(f)
    write(card([("AI Workout Coach", 2.0, (255, 255, 255)),
                ("exercise recognition - rep counting - form feedback", 0.9, (200, 200, 200)),
                ("all running live on a laptop CPU", 0.8, (160, 160, 160))], 3.5))
    results = []
    for name, f in DEMO_CLIPS.items():
        good = "good" in name
        write(card([(name.replace("·", "-"), 1.6, (113, 204, 46) if good else (60, 76, 231)),
                    ("the exercise is recognized automatically", 0.8, (180, 180, 180))]))
        pipe = CoachPipeline(app, rules, "auto")
        cap = cv2.VideoCapture(str(CLIPS / f))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n, written = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / src_fps
            n += 1
            r = pipe.process(frame, t)
            img = pipe.draw_panel(fit(pipe.annotate(frame, r, panel=False)), r)
            while written < int(t * FPS) + 1:  # garde la vitesse réelle à 30 fps
                wr.write(img)
                written += 1
        cap.release()
        s = pipe.summary()
        ex = max(s, key=lambda k: s[k]["reps"]) if s else None
        pipe.close()
        if ex:
            res = s[ex]
            results.append((name, res["reps"], res["avg_score"]))
            print(f"{name}: {res['reps']} reps, score {res['avg_score']:.0f}")
    lines = [("Summary", 1.6, (255, 255, 255))]
    for name, reps, score in results:
        color = (113, 204, 46) if score >= 75 else (18, 156, 243) if score >= 50 else (60, 76, 231)
        lines.append((f"{name.replace('·', '-')}:  {reps} reps,  form {score:.0f}/100", 0.8, color))
    write(card(lines, 5.0))
    wr.release()
    encode_h264(raw, out)
    print(f"-> {out} ({out.stat().st_size / 1e6:.0f} Mo)")


def encode_h264(raw: Path, out: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError:
        raw.replace(out)
        return
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw),
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out)], check=True)
    raw.unlink()


if __name__ == "__main__":
    main()
