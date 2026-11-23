# Next Steps: Better Models, a Web Interface and Any Camera

This is the plan for the next phase of the AI Workout Coach. It lists what to do, in what order, and how we will know it worked. The current state is in [README.md](../README.md).

## 1. Where We Stand (21 November 2026)

| Part | State | Main limit |
| --- | --- | --- |
| Pose (MediaPipe) | Works, 25-32 FPS on CPU | 2D angles depend on the camera view |
| Rep counter | 79% exact, 92% within ±1 rep (side and diagonal views) | Misses shallow reps; not tuned for the front view |
| Form rules (11 rules) | 72% of clips right (good/bad) | Thresholds tuned on side views only |
| Form decision + correction | Each rep: good or to fix, a short tip and a "how to correct it" text | Corrections only cover the 11 faults the rules can see |
| Random Forest form score | 74% of clips right (side views) | Scores good front-view squats 39-45, below the 50 needed for "good" |
| LSTM exercise classifier | 98% of clips right on unseen people; detects QEVD front-view squats at 100% confidence | Needs about 3.5 s before it decides |
| Interface | Streamlit, 3 pages | Camera is read by the server, so it only works on the same PC |
| Camera | Built-in or USB camera, chosen in the app ("Detect cameras") | Streamlit only, not in a browser yet |

**Measured on QEVD front-view squats (this week):** clean reps bottom out at a median knee angle of **89°**, shallow ones at **123°**. The current rule (`min_knee > 85` means too shallow) flags most good front-view squats. Of the 25 clips labelled shallow, 14 counted **0 reps**. The Random Forest has the same problem: it scores a clean front-view squat 39-45/100.

## 2. Step 0: Get the Data and Models Onto Every PC

Nothing else can be measured until this is done.

- [x] **Share the trained models.** Done on 23 November: `data/models/exercise_lstm.pt` and `data/models/form_model.joblib` are in Git LFS (`git lfs pull`). They load with scikit-learn 1.9.1 and torch 2.15 without version warnings.
- [ ] **Share the processed data.** `data/processed/all.parquet` and `reps.parquet` (Mendeley features) let anyone retrain without the 2.4 GB of videos. Put them in the same release or on a shared drive.
- [ ] **Complete QEVD-FIT-300k.** Each archive is the last piece of a split zip. With every `.z01` and `.z02` piece, `python ml/data/qevd_split_zip.py build data data/raw/qevd_fit300k` reads about **19,000** squat and push-up clips. Today: Part 1 `.z01` added, **1,175** clips (649 squats, 526 push-ups); still missing Part 1 `.z02` and Parts 2-4 `.z01`/`.z02`.
- [ ] **Pin the dependencies.** `pyproject.toml` only sets minimum versions, so every install gets different versions (this week: NumPy 2.5, pandas 3.1, torch 2.15). Add a lock file (`pip-compile` or `uv lock`) so everyone runs the same versions.
- [x] **Make the commands work on Windows.** `tasks.ps1` has the same tasks as the `Makefile` plus `retrain` and `use-candidates` (`.	asks.ps1` lists them).
- [x] **Repository organized by architecture** (`frontend/`, `backend/`, `ml/`, `data/`, `deploy/`, `docs/`, `tests/`; dependencies in `pyproject.toml`), 23 November.
- [x] **Training scripts read Mendeley + QEVD** (`coach/datasets.py`, `--sources`, `--out`) and report results per dataset and camera view. QEVD reps carry their named faults (`faults` column in `reps.parquet`), ready for per-fault models.
- [x] **GPU training on Windows:** CUDA build of PyTorch in `.venv` (the default pip install is CPU-only).

## 3. Better Models

### 3.1 Evaluation First

Every change below must be judged against a fixed test set; otherwise we cannot tell whether it helped.

- [ ] Freeze a test set: Mendeley subjects held out + a QEVD test split + **our own webcam clips** (`ml/data/record_clip.py`, step 8 of the README). At least 3 people, 3 exercises, good and bad form, side and front views.
- [ ] Hand-count reps on 100 clips, so the counter is no longer compared only with an automatic peak count.
- [ ] One script, one report: `ml/evaluation/evaluate_all.py` prints rep accuracy, form accuracy and per-fault precision/recall, **per exercise and per camera view**.

### 3.2 Camera View Awareness (largest expected gain)

The same squat measures 59° from the side and 92° from the front (median lowest knee angle). Thresholds must depend on the view.

- [x] Detect the view per rep from `shoulder_width` (`rep_view`, front if ≥ 0.55; 24 November). Still to do: calibrate the threshold on Mendeley's labelled front / side / diagonal clips.
- [x] One set of thresholds per view: each rule has `view: front|side` in `rules.yaml` (squat and push-up hips done). The original plan was:
  ```yaml
  - code: SQUAT_SHALLOW
    check: {side: "min_knee > 85", front: "min_knee > 105"}
  ```
  105° sits between the clean (89°) and shallow (123°) medians measured on QEVD; tune it on the full QEVD set.
- [ ] Try MediaPipe **world landmarks** (3D, in metres) for the angles. They depend much less on the camera view than the 2D image points we use now. Compare both on the test set before switching.

### 3.3 Rep Counter

- [ ] Count shallow reps instead of dropping them: QEVD shallow squats often gave 0 reps. Count a rep once the angle has moved by a minimum amount (for example 35°), then let the form rules mark it as too shallow.
- [ ] Front-view thresholds (from 3.2).
- [ ] Target: ≥ 90% exact on the frozen test set, all views.

### 3.4 Form Rules and Corrections

QEVD names the fault, not only good/bad. That lets us test each rule on its own fault.

- [ ] Tune each rule against its QEVD label (precision and recall per fault), not only against good/bad clips.
- [x] Squat, front view (24 November, QEVD test clips): back not straight 54% caught / 0% false alarms, feet too close 91% / 5%, knees caving in (5% false alarms), depth 53% / 7%. Knees over toes cannot be judged from the front; push-up rules still to do.
- [ ] Add rules for faults QEVD labels that we cannot see yet. The earlier test (`ml/evaluation/eval_qevd_squat.py`) showed they are learnable from our features:

  | Fault (QEVD label) | Earlier AUC | Feature to use | New rule |
  | --- | --- | --- | --- |
  | Narrow stance (`narrow`) | 0.95 | foot spread / shoulder width | `SQUAT_NARROW` |
  | Back not straight | 0.92 | torso lean, hip-shoulder line | `SQUAT_BACK` |
  | Shallow (`shallow`, `insufficient`) | 0.88 | knee angle | front-view threshold |
  | Knees over toes | 0.75 | knee forward | retune `SQUAT_KNEES_FORWARD` |
  | Knees caving in | not labelled | `knee_spread` | `SQUAT_VALGUS` |
  | Push-up hips too high (`elevated hips`) | not tested | body line | `PUSHUP_PIKE` |
  | Push-up head down | not tested | nose vs shoulder line | `PUSHUP_HEAD` |

- [ ] Each new rule gets a `tip` and a `fix` (how to correct it) in `rules.yaml`; a test already checks that every rule has a correction.
- [ ] Have a coach or physiotherapist review the correction texts once.

### 3.5 Form Score Model

- [ ] Replace the binary good/bad Random Forest with **one classifier per fault** (multi-label), trained on QEVD fault labels (already in `reps.parquet`) + Mendeley. The app then shows the probability of each fault, and the corrections are ranked by it.
- [ ] Try gradient boosting (LightGBM) and a small 1D-CNN or GRU on the rep's angle curve, not only on its min/max summary.
- [ ] Calibrate the probabilities (isotonic or Platt scaling) so that "score 70" means the same on every exercise.
- [ ] Watch for leakage: QEVD has no person ID, so the same person can be in train and test. Report QEVD results as optimistic and keep Mendeley subject-wise splits as the reference.

### 3.6 Exercise Classifier (LSTM)

- [x] Add QEVD squats and push-ups (front view) to training: done in `train_classifier.py`.
- [ ] Add QEVD's other exercises (lunges, jumping jacks, planks...) as `idle`: `qevd_split_zip.py build` only extracts squats and push-ups today.
- [ ] Add training augmentation already planned in `backend/coach/configs/model.yaml`: mirror, ±10° rotation, 0.8-1.2× speed.
- [ ] More exercises once the data allows: lunge and plank are both in QEVD.
- [ ] Export to ONNX as well as TorchScript (`torch.jit` is deprecated in the installed torch), so the model can also run in the browser (section 4).

### 3.7 Speed

- [ ] Benchmark lite / full / heavy pose models on our laptops (`ml/evaluation/benchmark_fps.py`). Use lite when FPS drops below 20.
- [ ] Target: ≥ 25 FPS on a mid-range laptop CPU with the full pipeline.

## 4. Web Interface

### 4.1 Why Move Beyond Streamlit

Streamlit is good for the demo, but:

- the camera is opened by the **server**, so the coach only works on the PC running the app, not from a phone or another laptop;
- each click reruns the whole page script, which makes a smooth live view hard;
- layout and mobile support are limited.

### 4.2 Recommended Architecture

Run the pose model **in the browser**, send only the 33 landmarks to the server.

```
Browser (any laptop / phone)                         Server (Python, FastAPI)
┌───────────────────────────────────────┐           ┌──────────────────────────────────┐
│ Camera (getUserMedia, chosen deviceId) │           │ Features -> LSTM -> rep counter   │
│ MediaPipe Pose Landmarker (JS, WASM)   │──────────►│ -> form rules -> feedback engine  │
│ Skeleton overlay on <canvas>           │ WebSocket │ SQLite / PostgreSQL               │
│ Reps, verdict, tip, "how to correct"   │◄──────────│ (same Python code as today)       │
└───────────────────────────────────────┘  results  └──────────────────────────────────┘
```

- **Privacy:** no video leaves the computer, only landmark numbers (about 1 KB per frame).
- **Same model:** MediaPipe's JavaScript version runs the same `pose_landmarker_full.task`. Check on 20 clips that the browser landmarks give the same angles as the Python ones before trusting the thresholds.
- **Reuse:** `compute_features`, `RepCounter`, `FormAnalyzer` and `FeedbackEngine` stay in Python unchanged; only the frame source changes.

Simpler alternative if time is short: keep Streamlit and add `streamlit-webrtc`, so the browser sends the camera to the server. It works from other devices, but video does leave the computer and latency is higher.

### 4.3 Pages

| Page | Content |
| --- | --- |
| Home | What the coach does, camera check, "Start" |
| Live coach | Video with skeleton, reps, last-rep verdict (good / to fix), tip, "How to correct it", form score |
| Session report | Reps per exercise, good vs to-fix reps, score per rep, corrections for each fault seen, best-practice checklist |
| History | Sessions over time, score trend, most frequent faults |
| Settings | Camera, exercise (auto or manual), voice on/off, units, language |
| Account (later) | Login, personal history, goals |

### 4.4 Features to Add

- [ ] Voice feedback with the browser's speech synthesis (the user is not looking at the screen during a squat).
- [ ] A short beep on each counted rep.
- [ ] Camera placement check before starting: whole body visible, side-on, enough light.
- [ ] Sets and rest timer, target reps per set.
- [ ] Export the session report as PDF.
- [ ] Phone-friendly layout (phone propped up as the camera).
- [ ] French and English texts (tips and corrections live in YAML, so this is a translation file).

### 4.5 Technology

| Layer | Choice |
| --- | --- |
| Frontend | React + Vite + TypeScript, `@mediapipe/tasks-vision` |
| Backend | FastAPI, WebSocket `/ws/session`, REST `/sessions`, `/sessions/{id}/report` |
| Database | SQLite now (SQLAlchemy), PostgreSQL when hosted (only `DATABASE_URL` changes) |
| Deployment | Docker Compose (already in the repo); HTTPS is required for camera access except on `localhost` |

## 5. Camera: Built-in or Plugged In

### 5.1 Done This Week (Streamlit App)

- `backend/coach/capture/camera.py`: `Camera` reads the webcam in a background thread and always gives the newest frame, so delay never builds up; `list_cameras()` finds the cameras that deliver an image.
- Live page: choosing **Webcam** lists the cameras ("Camera 0, usually the built-in camera", "Camera 1, usually a USB camera"). **Detect cameras** finds a camera plugged in after the app started.
- On Windows the DirectShow backend is used: detection takes about 1.4 s and lists USB webcams.
- Tested on this laptop: built-in camera found and read at 1280×720.

### 5.2 Still To Do

- [ ] Show the device **name** ("HD Webcam", "Logitech C920") instead of an index. OpenCV cannot read names; on Windows, `pygrabber` can, and in the browser `enumerateDevices()` gives them.
- [ ] Remember the last camera used.
- [ ] Warn when the camera is busy (Teams, Zoom, OBS) or unplugged during a session; today the session just stops.
- [ ] Use the camera's best resolution and frame rate up to 1280×720 at 30 FPS; low light lowers the frame rate, so suggest more light when FPS drops.
- [ ] Web version: `navigator.mediaDevices.enumerateDevices()` and `getUserMedia({video: {deviceId}})` to pick the camera in the browser.
- [ ] Phone as a camera: test DroidCam or Camo (they appear as a USB webcam).

### 5.3 Camera Placement Guide (to show in the app)

- Side-on to the camera for squats, push-ups and curls; front view works for squat depth and stance only.
- 2-3 m away, camera at hip height, whole body from head to feet in the picture.
- Light in front of you, not behind; plain background if possible.

## 6. Engineering

- [ ] GitHub Actions: run `pytest` and `ruff` on every push (39 tests today).
- [ ] Fix the warnings: `datetime.utcnow()` in the database models, `torch.jit` deprecation.
- [ ] Log FPS and pipeline timings per stage, to see where time goes on slow laptops.
- [ ] Keep `backend/coach/configs/` as the single place for thresholds, tips and corrections.

## 7. Suggested Order

| # | Milestone | Done when |
| --- | --- | --- |
| 1 | Data and models on every PC (section 2; models done) | A new PC runs the app with auto-detect in under 30 minutes |
| 2 | Frozen test set and evaluation report (3.1) | One command prints all metrics per exercise and view |
| 3 | View-aware thresholds and rep counter fixes (3.2, 3.3) | Front-view squat verdicts match QEVD labels; ≥ 90% exact reps |
| 4 | New rules and corrections from QEVD faults (3.4) | 5+ new faults detected, each with a tip and a correction |
| 5 | Per-fault form model (3.5) | Per-fault recall reported; better than the rules alone |
| 6 | Web app prototype (4.2, 5.2) | Live coach in the browser with camera choice, results over WebSocket |
| 7 | Full web interface (4.3, 4.4) | Report, history, voice feedback, phone layout |
| 8 | Validation with real users | 5 people, own cameras, written results |
