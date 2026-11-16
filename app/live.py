"""Page principale : le coach en direct, sur une vidéo de démo, une vidéo importée ou la webcam.

Page « Live coach » (lancée par app/streamlit_app.py).
"""
import tempfile
import time

import cv2
import streamlit as st

from coach.pipeline import CoachPipeline
from coach.ui import (
    CLIPS,
    DEMO_CLIPS,
    EXERCISE_LABELS,
    card,
    configs,
    render_report,
    repository,
    score_color,
    setup_page,
)

setup_page()
app_cfg, rules_cfg = configs()

st.title("🏋️ AI Workout Coach")
st.markdown('<div class="hero">Recognizes your exercise, counts your reps and corrects your form '
            '— in real time, entirely on this computer.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ réglages
with st.sidebar:
    st.header("Session")
    source = st.radio("Video source", ["Demo video", "Upload a video", "Webcam"])
    clip_path = None
    if source == "Demo video":
        label = st.selectbox("Demo clip", list(DEMO_CLIPS))
        clip_path = str(CLIPS / DEMO_CLIPS[label])
    elif source == "Upload a video":
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
        if up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + up.name.rsplit(".", 1)[-1]) as tmp:
                tmp.write(up.getbuffer())
            clip_path = tmp.name
    else:
        st.caption("Stand **side-on** to the camera, whole body in view.")
    mode_label = st.selectbox("Exercise", ["Auto-detect", "Squat", "Push-up", "Bicep curl"])
    mode = {"Auto-detect": "auto", "Squat": "squat", "Push-up": "pushup", "Bicep curl": "curl"}[mode_label]
    realtime = st.toggle("Play at real speed", value=True, disabled=source == "Webcam")
    c1, c2 = st.columns(2)
    start = c1.button("▶ Start", type="primary", width="stretch",
                      disabled=source != "Webcam" and clip_path is None)
    stop = c2.button("■ Stop", width="stretch")
    st.divider()
    st.caption("Tips: one at a time, safety first. Score: Random Forest trained on 26 people.")

if start:
    st.session_state.update(running=True, saved_id=None, result=None)
if stop:
    st.session_state.running = False

DISPLAY_H = 560  # hauteur d'affichage de la vidéo (les clips de profil sont souvent en portrait)
CANVAS_W = 760
PAGE_BG_BGR = (23, 17, 15)  # #0f1117, le fond du thème (.streamlit/config.toml)


def show_frame(img, caption: str | None = None) -> None:
    h, w = img.shape[:2]
    disp_w = int(w * DISPLAY_H / h)
    disp_w = min(disp_w, 760)  # vidéo paysage : on limite plutôt la largeur
    small = cv2.resize(img, (disp_w, int(h * disp_w / w)), interpolation=cv2.INTER_AREA)
    pad = (CANVAS_W - disp_w) // 2
    if pad > 0:  # vidéo portrait : centrée sur un fond de la couleur de la page
        small = cv2.copyMakeBorder(small, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=PAGE_BG_BGR)
    frame_box.image(small, channels="BGR", width="stretch", caption=caption)


# ------------------------------------------------------------------ mise en page
video_col, info_col = st.columns([3, 2], gap="large")
frame_box = video_col.empty()
with info_col:
    ex_box, reps_box, score_box = st.empty(), st.empty(), st.empty()
    tip_box = st.empty()
    log_box = st.empty()


def show_state(r, pipe) -> None:
    name = EXERCISE_LABELS.get(r.exercise, "Detecting…" if pipe.mode == "auto" else "–")
    conf = f"confidence {r.confidence:.0%}" if pipe.mode == "auto" and r.exercise else ""
    ex_box.markdown(card("Exercise", name, conf), unsafe_allow_html=True)
    reps_box.markdown(card("Reps", str(r.rep_count), f"{r.partial_count} partial"), unsafe_allow_html=True)
    sc = r.form_score
    score_box.markdown(card("Last rep form score",
                            f'<span style="color:{score_color(sc)}">{sc:.0f}</span>' if sc is not None else "–",
                            "out of 100"), unsafe_allow_html=True)
    if r.tip:
        good = any(w in r.tip for w in ("Great", "Nice", "Perfect"))
        tip_box.markdown(f'<div class="tip {"good" if good else "warn"}">{r.tip}</div>', unsafe_allow_html=True)
    else:
        tip_box.markdown('<div class="tip idle">Tips appear here after each rep</div>', unsafe_allow_html=True)


def run_session() -> None:
    pipe = CoachPipeline(app_cfg, rules_cfg, exercise=mode)
    is_cam = source == "Webcam"
    cap = cv2.VideoCapture(0 if is_cam else clip_path)
    if not cap.isOpened():
        st.error("Could not open the video source." + (" Is a webcam connected?" if is_cam else ""))
        st.session_state.running = False
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0, n = time.time(), 0
    try:
        while st.session_state.get("running"):
            ok, frame = cap.read()
            if not ok:
                break
            if is_cam:
                frame = cv2.flip(frame, 1)
                t = time.time() - t0
            else:
                t = n / fps
                if realtime and (wait := t - (time.time() - t0)) > 0:
                    time.sleep(wait)
            n += 1
            r = pipe.process(frame, t)
            img = pipe.annotate(frame, r, panel=False)
            show_frame(img)
            if n % 3 == 0 or r.new_rep:
                show_state(r, pipe)
            st.session_state.result = {"summary": pipe.summary(), "reps": pipe.log.reps,
                                       "duration": t}
    finally:
        cap.release()
        pipe.close()
    st.session_state.running = False


if st.session_state.get("running"):
    run_session()

# ------------------------------------------------------------------ bilan
res = st.session_state.get("result")
if res and not st.session_state.get("running"):
    if res["reps"] and not st.session_state.get("saved_id"):
        st.session_state.saved_id = repository().save_pipeline_session(res["reps"], res["duration"])
    st.divider()
    st.header("Session report")
    if st.session_state.get("saved_id"):
        render_report(repository().session_report(st.session_state.saved_id))
        st.caption("Saved to the session history (see the **History** page).")
    else:
        st.info("No complete rep was detected in this session.")
elif not st.session_state.get("running"):
    # Écran d'accueil, dessiné dans les mêmes emplacements que la vue live (rien ne reste à l'écran).
    preview = None
    if clip_path:
        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ok, preview = cap.read()
        cap.release()
    if preview is not None:
        show_frame(preview, caption="Press ▶ Start in the sidebar")
    else:
        frame_box.info("Choose a source in the sidebar and press **▶ Start**.")
    ex_box.markdown(card("How it works", "4 stages",
                         "1 · Pose: MediaPipe finds 33 body points<br>"
                         "2 · Exercise recognition: LSTM network<br>"
                         "3 · Rep counting: state machine on joint angles<br>"
                         "4 · Form check: rules + Random Forest score"), unsafe_allow_html=True)
    reps_box.markdown(card("Tip", "Stand side-on",
                           "The whole body must be visible. Side view works best for push-ups and curls."),
                      unsafe_allow_html=True)
