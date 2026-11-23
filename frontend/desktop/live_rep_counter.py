"""Démo du compteur de répétitions et des conseils de posture (sans ML) :
webcam ou vidéo -> squelette + compteur + un conseil et un score après chaque rep.

Le choix de l'exercice est manuel ici (le classifieur LSTM viendra à l'étape 6).
Placez-vous DE PROFIL face à la caméra : c'est la vue où la détection est fiable.

Usage :
    python frontend/desktop/live_rep_counter.py --exercise squat              # webcam 0
    python frontend/desktop/live_rep_counter.py --exercise curl --source data/raw/fitness_form/subject_001_bicep_curl_good_side.mp4
    python frontend/desktop/live_rep_counter.py --exercise squat --source video.mp4 --no-display   # test sans fenêtre
Touches : q = quitter, r = remettre à zéro.
"""
import argparse
import time

import cv2

from coach.config import load_config
from coach.features.angles import compute_features
from coach.features.smoothing import FeatureSmoother, JumpFilter
from coach.pose.estimator import PoseEstimator
from coach.pose.skeleton import draw_skeleton
from coach.pose.skeleton import put_text as text
from coach.reasoning.form_analyzer import FormAnalyzer
from coach.reasoning.rep_counter import RepCounter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", choices=["squat", "pushup", "curl"], required=True)
    ap.add_argument("--source", default="0", help="index de webcam ou chemin d'une vidéo")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    rules, app = load_config("rules"), load_config("app")
    th = rules["rep_counter"][args.exercise]

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, app["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app["camera"]["height"])
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source {args.source}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    est = PoseEstimator(min_visibility=app["pose"]["min_visibility"])
    jf, sm = JumpFilter(), FeatureSmoother()
    counter = RepCounter(args.exercise, th, fps=fps_src)
    analyzer = FormAnalyzer(rules)
    scores, fault_counts = [], {}
    last_msg, n, t0 = "", 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_cam and app["camera"]["mirror"]:
            frame = cv2.flip(frame, 1)
        t = (time.time() - t0) if is_cam else n / fps_src
        n += 1
        h, w = frame.shape[:2]
        lm = est.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms=int(t * 1000))
        angle = None
        if lm is None or not est.key_joints_visible(lm):
            last_msg = "Step back so I can see you"
        else:
            f = sm(jf(compute_features(lm, aspect=w / h)), t)
            angle = f[th["angle"]]
            ev = counter.update(f, timestamp=t)
            if ev:
                faults = analyzer.check_rules(args.exercise, ev)
                score = analyzer.form_score(ev, faults)
                head = f"Rep {ev['rep_index']}" if ev["counted"] else "Partial rep"
                tip = faults[0].tip if faults else ("Good rep!" if ev["counted"] else "Go further")
                last_msg = f"{head} ({score:.0f}/100): {tip}"
                if ev["counted"]:
                    scores.append(score)
                for flt in faults:
                    fault_counts[flt.code] = fault_counts.get(flt.code, 0) + 1
                if args.no_display:
                    print(last_msg)
            draw_skeleton(frame, lm)

        if not args.no_display:
            s = h / 720
            text(frame, f"{args.exercise.upper()}  reps: {counter.count}", (20, int(60 * s)), 1.6 * s, (0, 255, 0))
            text(frame, f"partial: {counter.partial_count}   state: {counter.state.value}", (20, int(110 * s)), 0.9 * s)
            if angle is not None:
                text(frame, f"{th['angle']}: {angle:.0f} deg  (top>{counter.effective_top:.0f}, bottom<{th['bottom_below']})",
                     (20, int(150 * s)), 0.8 * s, (0, 255, 255))
            text(frame, last_msg, (20, h - int(30 * s)), 1.2 * s, (0, 200, 255))
            cv2.imshow("AI Workout Coach - rep counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                counter.reset()
                scores.clear()
                fault_counts.clear()

    cap.release()
    est.close()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"Reps: {counter.count}  partial: {counter.partial_count}  "
          f"({n} frames, {n / max(elapsed, 1e-6):.1f} FPS de traitement)")
    if scores:
        print(f"Score moyen : {sum(scores) / len(scores):.0f}/100   défauts : {fault_counts or 'aucun'}")


if __name__ == "__main__":
    main()
