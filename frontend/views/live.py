"""Page principale : le coach en direct, sur une vidéo de démo, une vidéo importée ou la webcam.

Page « Live coach » (lancée par frontend/streamlit_app.py).
"""
import tempfile
import time

import cv2
import streamlit as st

from coach.capture.camera import Camera, list_cameras
from coach.demo_clips import DEMO_CLIPS
from coach.pipeline import CoachPipeline
from ui import (
    EXERCISE_LABELS,
    classifier_available,
    coach_panel,
    configs,
    render_report,
    repository,
    setup_page,
    voice,
)

setup_page()
app_cfg, rules_cfg = configs()

st.title("Workout Coach")
st.markdown('<p class="muted">Counts your reps, checks your form and tells you how to fix it. '
            'Everything runs on this computer.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ réglages
with st.sidebar:
    st.header("Session")
    source = st.radio("Video source", ["Demo video", "Upload a video", "Webcam"])
    clip_path = None
    if source == "Demo video":
        if DEMO_CLIPS:
            label = st.selectbox("Demo clip", list(DEMO_CLIPS))
            clip_path = str(DEMO_CLIPS[label])
        else:
            st.warning("No demo clip on this PC. Run `python ml/data/qevd_split_zip.py build data "
                       "data/raw/qevd_fit300k` or upload a video.")
    elif source == "Upload a video":
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
        if up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + up.name.rsplit(".", 1)[-1]) as tmp:
                tmp.write(up.getbuffer())
            clip_path = tmp.name
    else:
        refresh = st.button("Detect cameras", width="stretch")
        if refresh or "cameras" not in st.session_state:
            with st.spinner("Looking for cameras"):
                st.session_state.cameras = list_cameras()
        cams = st.session_state.cameras
        if cams:
            cam_label = st.selectbox("Camera", [c["label"] for c in cams],
                                     help="Built-in laptop camera or a webcam plugged in by USB. "
                                          "Plugged a camera in? Press Detect cameras.")
            cam_index = next(c["index"] for c in cams if c["label"] == cam_label)
        else:
            cam_index = None
            st.error("No camera found. Plug one in, close other apps using it (Teams, Zoom) "
                     "and press Detect cameras.")
        st.caption("Stand **side-on** to the camera, whole body in view.")
    has_lstm = classifier_available()
    mode_label = st.selectbox("Exercise", (["Auto-detect"] if has_lstm else []) + ["Squat", "Push-up", "Bicep curl"])
    if not has_lstm:
        st.caption("Auto-detect needs the trained LSTM (`data/models/exercise_lstm.pt`): pick the exercise.")
    mode = {"Auto-detect": "auto", "Squat": "squat", "Push-up": "pushup", "Bicep curl": "curl"}[mode_label]
    realtime = st.toggle("Play at real speed", value=True, disabled=source == "Webcam")
    voice_on = st.toggle("Voice coaching", value=True,
                         help="Reads each correction out loud (\"Go deeper\", \"Pin your elbows\") "
                              "and says \"Great, keep going\" after 2 good reps.")
    if voice_on and not voice().available:
        st.caption("No speech engine found on this computer: corrections are shown on screen only.")
    c1, c2 = st.columns(2)
    start = c1.button("Start", type="primary", width="stretch",
                      disabled=clip_path is None if source != "Webcam" else cam_index is None)
    stop = c2.button("Stop", width="stretch")

if start:
    st.session_state.update(running=True, saved_id=None, result=None)
if stop:
    st.session_state.running = False

DISPLAY_H = 560  # hauteur d'affichage de la vidéo (les clips de profil sont souvent en portrait)
CANVAS_W = 760
PAGE_BG_BGR = (239, 244, 245)  # Paper #F5F4EF, le fond du thème (.streamlit/config.toml)


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
panel_box = info_col.empty()


def show_state(r, pipe) -> None:
    name = EXERCISE_LABELS.get(r.exercise, "Detecting" if pipe.mode == "auto" else "-")
    sub = f"{r.partial_count} partial"
    if pipe.mode == "auto" and r.exercise:
        sub += f" · recognised at {r.confidence:.0%}"
    good = bool(r.tip) and any(w in r.tip for w in ("Great", "Nice", "Perfect"))
    panel_box.markdown(coach_panel(name, str(r.rep_count), sub, r.verdict, r.tip, "good" if good else "warn",
                                   r.fix, r.form_score), unsafe_allow_html=True)


def frames(is_cam: bool):
    """Frames BGR et leur instant t (s). Webcam : thread de capture, toujours la frame la plus récente."""
    if is_cam:
        c = app_cfg["camera"]
        with Camera(cam_index, c["width"], c["height"], c["fps"], c["mirror"]) as cam:
            t0, seq = time.time(), -1
            while cam.running:
                frame, t, seq = cam.latest_frame(after_seq=seq, timeout=5.0 if seq < 0 else 2.0)
                if frame is None:
                    break
                yield frame, t - t0
        return
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0, n = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps
            if realtime and (wait := t - (time.time() - t0)) > 0:
                time.sleep(wait)
            n += 1
            yield frame, t
    finally:
        cap.release()


def run_session() -> None:
    pipe = CoachPipeline(app_cfg, rules_cfg, exercise=mode)
    is_cam = source == "Webcam"
    n, t = 0, 0.0
    source_frames = frames(is_cam)
    try:
        for frame, t in source_frames:
            if not st.session_state.get("running"):
                break
            n += 1
            r = pipe.process(frame, t)
            if r.say and voice_on:
                voice().say(r.say)  # consigne courte, lue sans bloquer la vidéo
            img = pipe.annotate(frame, r, panel=False)
            show_frame(img)
            if n % 3 == 0 or r.new_rep:
                show_state(r, pipe)
            st.session_state.result = {"summary": pipe.summary(), "reps": pipe.log.reps,
                                       "duration": t}
    except RuntimeError as e:
        st.error(str(e))
    finally:
        source_frames.close()  # libère la caméra, même si on quitte la boucle (bouton Stop)
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
        st.caption("Saved to your session history (History page).")
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
        show_frame(preview, caption="Press Start in the sidebar")
    else:
        frame_box.info("Choose a video source in the sidebar, then press Start.")
    panel_box.markdown(coach_panel(mode_label, "0", "Stand side-on, 2 to 3 m from the camera, "
                                   "with your whole body in view."), unsafe_allow_html=True)
