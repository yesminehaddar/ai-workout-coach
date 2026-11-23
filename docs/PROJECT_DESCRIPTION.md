# AI Workout Coach: full project description

*Status as of 2026-11-17. Steps 1 to 7 of 8 are done, plus a visual quality check of the detections (step 3b). Anything marked **planned** is designed but not yet coded.*

## 1. The idea

AI Workout Coach is a personal trainer that runs on a laptop with a webcam. The user stands in front of the camera and exercises. The program:

1. **recognizes which exercise** is being done (squat, push-up or bicep curl),
2. **counts the repetitions** automatically,
3. **judges the quality of the posture** on each repetition,
4. **gives one short spoken or written tip at a time** ("Go a bit lower", "Chest up!"),
5. **saves the session** so progress can be reviewed later.

Everything runs locally, in real time (target: 20+ frames per second on a normal CPU, about 50 ms per frame). No video leaves the computer.

**Why it is interesting.** Most fitness apps only count reps. A coach that notices *how* you move (depth, back angle, elbow drift) and explains the fix is more useful, and it also helps prevent injuries from bad form.

**Who it is for.** Someone training at home without a coach. It is also a complete computer-vision and machine-learning project: pose estimation, time-series classification, a rule engine, a database and a web interface.

## 2. How it works (the pipeline)

Each frame goes through seven stages:

```
Webcam ─► 1 Capture ─► 2 Pose (33 points) ─► 3 Features (angles, smoothing, 30-frame window)
                                                 │
            ┌────────────────────────────────────┼─────────────────────────────┐
            ▼                                    ▼                             ▼
   4 Exercise classifier (LSTM)        5 Rep counter (state machine)   6 Form analyzer (rules + Random Forest)
            └────────────────────────────────────┴──────────────┬──────────────┘
                                                                 ▼
                                              7 Feedback engine (one tip at a time) ─► Streamlit UI
                                                                 │
                                                                 ▼
                                                       SQLite database
```

### Stage 1: Capture (OpenCV)
Reads the webcam (mirrored like a mirror) or a video file with OpenCV. *(done, inside the app)*

### Stage 2: Pose estimation (MediaPipe) *(done)*
MediaPipe's `PoseLandmarker` finds **33 body landmarks** per frame (nose, shoulders, elbows, wrists, hips, knees, ankles, and so on), each with x, y, z and a visibility score.
- It runs in video mode, which tracks the person across frames and is smoother than detecting each frame alone.
- `key_joints_visible()` checks that at least one full side of the body is visible. If not, the coach pauses and says "Step back so I can see you".
- Only x and y are used. The depth value z is too noisy.

### Stage 3: Features *(done)*
Landmarks are turned into **16 numbers per frame** (`coach/features/angles.py`). Distances are divided by torso length, so the result does not depend on how far the user stands. The better-visible side of the body (left or right) is used.

| Group | Features |
|---|---|
| Joint angles | `knee`, `hip`, `elbow`, `shoulder` |
| Posture | `torso_lean` (0 = upright), `body_line` (180 = straight, for push-ups) |
| Positions (normalized) | `wrist_height`, `hip_height`, `elbow_height`, `knee_forward`, `wrist_forward` |
| Shape ratios | `shoulder_width` (also hints at the camera view), `knee_spread` (below 1 = knees caving in), `elbow_flare` |
| Bookkeeping | `visibility`, `side` (0 = left, 1 = right) |

- **One Euro filter:** removes landmark jitter without adding lag. It smooths strongly when the body is still and lightly when it moves fast.
- **Sliding window:** keeps the last 30 frames (2 seconds, sampled at 15 frames per second) as a `(30, 16)` array for the classifier.

### Stage 4: Exercise classifier (LSTM) *(done)*
**Result** (one fixed split, not a cross-validation; `ml/training/train_classifier.py`, 3 minutes on the RTX 3050): tested on 5 subjects never seen in training, **96.7% of 2-second windows** are right (push-up 100%, curl 97%, squat 93%) and **98% of clips** by majority vote. The dataset has no idle footage, so its 4 other exercises (shoulder press, tricep, back row, abs) stand in for `idle`. Validation for early stopping uses 3 other subjects. Normalization is stored inside the model.
A small PyTorch LSTM (2 layers × 64 units, dropout 0.3) reads the 30-frame window and outputs one of `squat | pushup | curl | idle`.
- It runs every 5 frames, not every frame.
- The label only changes after **5 consecutive votes above 70% confidence**, which stops it flickering between exercises.
- Trained on the GPU; exported as TorchScript for the app.

### Stage 5: Rep counter (state machine) *(done)*
Follows the movement through four states: `TOP → GOING_DOWN → BOTTOM → GOING_UP → TOP`. Returning to TOP is **+1 rep**. If the user turns back before reaching BOTTOM, it counts as a **partial rep** and is not counted. It needs no machine learning.
- Nothing is counted until the user has been seen in the starting position.
- A **glitch filter** first discards frames where an angle jumps by more than 40° (a detection error), unless the jump lasts more than 3 frames.
- Both thresholds **adapt to the user** (fixed 15 Nov 2026). The 2D angle depends on the camera view: from the diagonal, a full push-up can look like 72–110° and a deep squat like 120°, so fixed thresholds missed most reps. When the angle has moved by at least 25° in the last 5 seconds, a rep counts once the angle drops past the midpoint of that person's own range (5th to 95th percentile, so one bad frame cannot skew it), and the top sits 15% of the range below their highest angle. Below 25° of movement the fixed thresholds apply. A shallow rep is therefore counted, and the form analyzer says "go lower". The first reps are re-judged when they end, once the calibration has seen them.
- Each finished rep returns a summary (min and max of every feature, tempo) that the form analyzer will use.

Thresholds were set from the real angle curves (`backend/coach/configs/rules.yaml`):

| Exercise | Angle used | Top (above) | Bottom (below) | Typical lowest angle in the data |
|---|---|---|---|---|
| Squat | knee | 160° | 100° | 66–72° |
| Push-up | elbow | 150° | 100° | 72–79° |
| Curl | elbow | 145° | 80° | 57–65° |

**Evaluation** (`ml/evaluation/eval_rep_counter.py`, 296 clips, side and diagonal views), on the reps actually shown to the user. The reference is an offline peak count of each angle curve, not a hand count. **79% of clips exact, 92% within ±1 rep**, on the same clips the thresholds were tuned on, so this is optimistic. Side view: curl 100%, squat 96%, push-up 88%; diagonal: curl 100%, push-up 90%, squat 78%. *Correction:* the 88% previously reported here counted partial reps as reps; on counted reps alone the old counter was only 58% exact and 70% within ±1. A rough check by eye on contact sheets (7 disputed clips opened, about 5 countable) went both ways: on one clip of about 16 push-ups the counter said 16 and the reference 2, but the counter still under-counted two others (about 8 real reps counted as 4, about 14 shallow reps counted as 10). This is not a hand-counted test set. Squats where the knees barely bend (about 25° of movement) are still not counted.

**Live demo:** `python frontend/desktop/live_rep_counter.py --exercise squat` (or `pushup`, `curl`) opens the webcam and shows the skeleton, the count and the state. It runs at 32–41 FPS on the CPU.

### Stage 6: Form analyzer (hybrid) *(done)*
- **Rules** (explainable, `coach/reasoning/form_analyzer.py`): each fault has a code, a tip and a priority (1 safety, 2 range of motion, 3 tempo or placement). A rule is a line in `backend/coach/configs/rules.yaml` such as `min_knee > 85`, checked on the summary of each rep: the min, max and range of every feature during the rep, and its duration.

| Exercise | Fault | Check | Tip | Fires on good / bad reps |
|---|---|---|---|---|
| Squat | `SQUAT_LEAN` | torso lean > 55° | Chest up! | 7% / 5% |
| | `SQUAT_SHALLOW` | lowest knee angle > 85° | Go a bit lower | 8% / 25% |
| | `SQUAT_KNEES_FORWARD` | knee > 0.52 torso ahead of ankle | Sit back, knees behind toes | 5% / 16% |
| | `SQUAT_FAST` | rep < 1.07 s | Slow down, control the descent | 6% / 25% |
| Push-up | `PUSHUP_SAG` | body line < 148° | Tighten your core, hips up | 6% / 28% |
| | `PUSHUP_SHALLOW` | lowest elbow angle > 100° | Lower your chest | 9% / 25% |
| | `PUSHUP_HANDS_FORWARD` | wrist ahead of shoulder | Hands under your shoulders | 5% / 15% |
| Curl | `CURL_BODY_SWING` | torso lean varies > 6.5° | Don't swing your body | 6% / 13% |
| | `CURL_SWING` | shoulder angle varies > 19° | Keep elbows still | 5% / 18% |
| | `CURL_PARTIAL` | lowest elbow angle > 70° | Squeeze to the top | 7% / 30% |
| | `CURL_FAST` | rep < 1.0 s | Slow down, control the weight | 4% / 36% |

- **How the thresholds were set.** `ml/data/extract_rep_summaries.py` cuts every clip into reps (4,620 reps from 434 clips, `data/processed/reps.parquet`). Good and bad clips of the same person were compared to find what actually differs. Push-up: hips sag and hands sit further forward in about 85% of subjects. Curl: shorter range and faster reps. Squat: "bad" means different things for different people, so no single measure separates it. Each threshold was then set so the rule fires on only 5–9% of good reps. The dataset never says *which* fault a bad clip has, so this is the best check available.
- **Result** (`ml/evaluation/eval_form_rules.py`, reliable views only). A clip is judged bad when more than 30% of its reps have a fault. **72% of 244 clips are classified correctly**: curl 76%, push-up 71%, squat 65%. The thresholds were tuned on the same clips, so these figures are a little optimistic.
- Rules also give a temporary **score**: 100 minus 30 per safety fault, 20 per range fault and 10 per tempo fault.
- **Random Forest** (200 trees, one per exercise, `ml/training/train_form_model.py`): takes the summary of one rep and returns a **form score from 0 to 100** (probability of "good"), replacing the temporary score. Cross-validated by subject (5 folds): **74% of clips right** (curl 82%, squat 71%, push-up 68%), slightly above the rules, and this time on people the model never saw. The rules say *what* to fix. The model says *how good* the rep was overall.

### Stage 7: Feedback engine *(done)*
Chooses **one tip at a time** so the user is not overwhelmed (`coach/feedback/engine.py`).
- Priority: safety (back, knees) before depth before tempo.
- Each tip has a 3-second cooldown.
- After 3 clean reps in a row, it encourages the user.
- If no rule fires but the Random Forest score is below 50, it gives a general tip ("Slow down and control the full movement") instead of praise: the rules do not see every mistake.
- Every tip shown is stored for the report.

### The pipeline *(done)*
`coach/pipeline.py` runs all the stages on each frame at **25–32 FPS on a CPU**. In auto mode, the LSTM needs about 3 seconds to recognize the exercise; a rep counter runs in parallel for each exercise meanwhile, so those first reps are kept once the exercise is known. On the six demo clips it recognizes every exercise and matches the automatic rep count, with scores of 90–96 for good form and 2–5 for bad form. Those clips were hand-picked for matching counts and clear contrast, and were used in training: they illustrate the app and do not measure it.

## 3. The interface (Streamlit) *(done)*
Run with `python -m streamlit run frontend/streamlit_app.py`. Presentation guide: [DEMO.md](DEMO.md).
- **Live coach:** a demo video, an uploaded video or the webcam, with the skeleton drawn on it, the detected exercise, the rep counter, the last rep's form score and the current tip. The webcam is read with OpenCV, so this mode needs the app to run locally (not in Docker).
- **Session report:** totals, form score per rep and the most frequent mistakes, read back from the database.
- **History:** every session, with the average score over time.
- **Backup demo video:** `frontend/desktop/make_demo_video.py` writes `data/demo/coach_demo.mp4` (3 min, H.264): six clips, good and bad form for each exercise, with the full overlay and a summary.

## 4. Data storage (SQLite) *(done)*
Five linked tables, defined with SQLAlchemy in `coach/db/models.py`:

```
users ─► sessions ─► sets ─► reps ─► form_events
```
- **users:** name, creation date.
- **sessions:** start time, duration, average score.
- **sets:** exercise, rep count, partial count.
- **reps:** min and max angle, tempo, form score, whether it was counted.
- **form_events:** fault code, tip text, time shown.

## 5. Training data

**Main dataset:** Mendeley "fitness exercises, good and bad form" (CC BY 4.0).
- 1,042 videos, 2.4 GB, 26 subjects, 7 exercises: squat, push-up, bicep curl, shoulder press, tricep, back row, abs (side bend).
- Each exercise is filmed with **correct and incorrect form**, from **3 views** (front, side, diagonal).
- Filename pattern: `subject_XXX_<exercise>_<good|bad>_<view>.mp4`.
- About 30 seconds each at 15 fps. Resolution is **not uniform**: 717 videos are 810×1440, but others range from 348×624 up to 1620×2880, and a few are landscape. Low-resolution clips (about 360×640, roughly 15% of the set) give less reliable landmarks.

**Extracted features (done).** `ml/data/extract_keypoints.py` ran MediaPipe over every video (61 minutes, 6 CPU cores). Output: `data/processed/all.parquet` with **445,208 frames from 1,041 videos**, pose detected in 99.9% of frames. Each frame keeps the labels, the 16 features, and the raw 33 landmarks, so new features can be computed later without rerunning MediaPipe.

**Known data issues**
- One file (`subject_024_back_good_diagonal.mp4`) is a corrupt 404-byte download and is excluded.
- Two tricep side-view videos have weak pose detection (52% and 63%). They will be filtered out.

**Visual check (done).** `ml/data/render_keypoints.py` draws the skeleton on the videos and writes `data/keypoint_previews/`: 39 annotated clips, an `overview.jpg`, and `quality_report.csv` covering every video. A frame is marked "glitch" when an angle jumps by more than 40° from the previous frame. Results: 220 of 1,041 clips (21%) are flagged, concentrated in specific exercise and view combinations:

| Exercise | Flagged, front | Flagged, side | Flagged, diagonal |
|---|---|---|---|
| Push-up | 98% | 6% | 6% |
| Bicep curl | 52% | 0% | 6% |
| Squat | 16% | 8% | 4% |

Decision: front-view push-up and curl clips are dropped from rep counting and training. Thresholds are tuned on side-view clips (and diagonal for push-up). The live app will ask the user to stand side-on to the camera.

**Penn Action** (squats and push-ups with annotated body points) is an optional extra and is **not installed**.

## 6. Key design decisions

- **Split by person, never by clip.** With only 26 subjects, putting clips of the same person in both training and test sets would inflate the results.
- **Rep counting without ML.** Angle thresholds are simpler, explainable, and already useful on their own.
- **Hybrid form analysis.** Rules give explainable tips; the Random Forest gives an overall score.
- **Training and live features must match.** The same MediaPipe model and the same feature code are used offline and in the app.
- **CPU for pose, GPU for training.** MediaPipe has no GPU support on Windows. PyTorch uses the RTX 3050 for the LSTM.
- **Data augmentation:** mirroring, ±10° rotation, speed 0.8× to 1.2×.
- **Everything configurable in YAML** (thresholds, tips, model settings), so tuning needs no code change.

## 7. Tech stack

| Area | Tools |
|---|---|
| Vision | OpenCV, MediaPipe Pose Landmarker |
| ML | PyTorch (LSTM), scikit-learn (Random Forest), joblib |
| Data | pandas, Parquet (pyarrow), NumPy |
| Storage | SQLite via SQLAlchemy |
| Interface | Streamlit (webcam through OpenCV) |
| Experiment tracking | MLflow (port 5000) |
| Packaging | Docker and docker-compose (app on port 8501), Makefile, Git LFS for model files |

## 8. Project layout

```
ai-workout-coach/
├── frontend/                 What the user sees
│   ├── streamlit_app.py      web app entry point (navigation between the views)
│   ├── views/                live coach, session report, history
│   ├── ui.py                 shared look: cards, report, best-practice texts
│   └── desktop/              OpenCV window demo (live_rep_counter.py), backup demo video
├── backend/                  The coach engine (Python package `coach`, used by the frontend)
│   └── coach/
│       ├── capture/          camera: built-in or USB, background capture thread
│       ├── pose/             MediaPipe pose landmarker, skeleton drawing
│       ├── features/         joint angles, One Euro smoothing, 30-frame window
│       ├── reasoning/        exercise classifier (LSTM), rep counter, form analyzer
│       ├── feedback/         one tip at a time + how to correct it
│       ├── db/               SQLite sessions, sets, reps, form events
│       ├── configs/          app.yaml, rules.yaml (thresholds, tips, corrections), model.yaml
│       ├── pipeline.py       runs every stage on each frame
│       └── datasets.py       Mendeley + QEVD loading for training
├── ml/                       Offline machine learning
│   ├── data/                 dataset tools: QEVD split-zip reader, keypoint extraction, rep summaries
│   ├── training/             train the LSTM classifier and the Random Forest form model
│   └── evaluation/           rep counter, form rules, QEVD faults, FPS benchmark
├── data/                     Not in Git, except the models (Git LFS)
│   ├── models/               pose model, exercise_lstm.pt, form_model.joblib
│   ├── raw/  processed/      videos, extracted features (Parquet)
│   └── db/  mlruns/  demo/   local database, training runs, demo video
├── deploy/                   Dockerfile, docker-compose.yml, .env.example
├── docs/                     architecture, demo guide, project description, next steps
├── tests/                    unit tests
├── pyproject.toml            package, dependencies (pip install -e ".[dev]"), pytest and ruff settings
├── Makefile, tasks.ps1       common commands (tasks.ps1 for Windows)
└── README.md, LICENSE
```
The slide decks are kept outside this repository.

## 9. Progress

| # | Step | Status |
|---|---|---|
| 1 | Setup: packages, pose model, test runner | ✅ Done |
| 2 | Pose estimator, features, smoothing, window, tests | ✅ Done |
| 3 | Offline extraction of all videos to Parquet | ✅ Done |
| 3b | Visual check of the detections: annotated clips, overview image, quality report for all videos | ✅ Done |
| 4 | Rep counter: glitch filter, thresholds tuned on real curves, state machine with thresholds that adapt to the user and the camera view (79% exact, 92% within ±1), live demo script | ✅ Done |
| 5 | Form analyzer: rules tuned and checked against the good/bad labels (72% of clips), tips and score in the live demo | ✅ Done |
| 6 | Train the LSTM classifier (96.7% of windows, 98% of clips) and Random Forest (74% of clips), split by subject, MLflow tracking | ✅ Done |
| 7 | Runtime: feedback engine, pipeline, database access, Streamlit pages, backup demo video | ✅ Done |
| 8 | **Validate on the user's own webcam clips** | ⏭ Next |

**First usable milestone:** the end of step 4, a webcam rep counter with no ML.

## 10. Risks and open questions

- **Small number of subjects (26).** The model may look good on the dataset and fail on a new person. Mitigation: subject-wise splits, and testing on the user's own clips (step 8).
- **Dataset vs. real webcam.** The clips are controlled recordings; a home webcam has different lighting, framing and angle.
- **Good/bad labels are binary.** The dataset does not say *what* was wrong, so specific tips depend on the hand-written rules.
- **Camera view.** Angles are less reliable from the front and diagonal views, and most reliable from the side. Squat knee-angle minimums confirm it: median 59° (side), 92° (front), 83° (diagonal). The `shoulder_width` feature gives the models a hint about the view. Thresholds should be tuned on the side view first.
- **"Pose detected" does not mean "pose correct".** 99.9% of frames had a detection, but some detections are wrong, mostly on low-resolution clips. Quality filters (visibility, resolution) are needed before training.
- **Real-time speed.** The 20+ FPS target has not been measured yet (`ml/evaluation/benchmark_fps.py` is a stub).
