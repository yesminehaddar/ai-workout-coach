# AI Workout Coach: full project description

*Status as of 2026-11-09. Steps 1 to 4 of 8 are done, plus a visual quality check of the detections (step 3b). Anything marked **planned** is designed but not yet coded.*

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
Reads the webcam (1280×720, 30 fps, mirrored like a mirror). Settings are in `configs/app.yaml`. *Planned.*

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
- **Sliding window:** keeps the last 30 frames (about 1 second) as a `(30, 16)` array for the classifier.

### Stage 4: Exercise classifier (LSTM) *(planned)*
A small PyTorch LSTM (2 layers × 64 units, dropout 0.3) reads the 30-frame window and outputs one of `squat | pushup | curl | idle`.
- It runs every 10 frames, not every frame.
- The label only changes after **5 consecutive votes above 70% confidence**, which stops it flickering between exercises.
- Trained on the GPU; exported as TorchScript for the app.

### Stage 5: Rep counter (state machine) *(done)*
Follows the movement through four states: `TOP → GOING_DOWN → BOTTOM → GOING_UP → TOP`. Returning to TOP is **+1 rep**. If the user turns back before reaching BOTTOM, it counts as a **partial rep** and is not counted. It needs no machine learning.
- Nothing is counted until the user has been seen in the starting position.
- A **glitch filter** first discards frames where an angle jumps by more than 40° (a detection error), unless the jump lasts more than 3 frames.
- The **top threshold calibrates itself** to the user: if their arms or legs never straighten to the configured value, it drops to 10° below the highest angle they reached in the last 5 seconds. The bottom threshold stays fixed, so shallow reps are still flagged.
- Each finished rep returns a summary (min and max of every feature, tempo) that the form analyzer will use.

Thresholds were set from the real angle curves (`configs/rules.yaml`):

| Exercise | Angle used | Top (above) | Bottom (below) | Typical lowest angle in the data |
|---|---|---|---|---|
| Squat | knee | 160° | 100° | 66–72° |
| Push-up | elbow | 150° | 100° | 72–79° |
| Curl | elbow | 145° | 80° | 57–65° |

**Evaluation** (`scripts/eval_rep_counter.py`, 296 clips, side and diagonal views). The dataset has no hand counts, so the counter is compared with an independent offline peak count of each angle curve. **88% of clips are within ±1 movement.** Side view: curl 100%, squat 88%, push-up 75%. From the diagonal view, many squats come out as "partial" because the camera angle makes the knee look less bent.

**Live demo:** `python scripts/live_rep_counter.py --exercise squat` (or `pushup`, `curl`) opens the webcam and shows the skeleton, the count and the state. It runs at 32–41 FPS on the CPU.

### Stage 6: Form analyzer (hybrid) *(planned)*
- **Rules** (explainable): each fault has a code, a tip and a priority.
  - Squat: `SQUAT_SHALLOW` (not deep enough), `SQUAT_LEAN` (torso leaning more than 45°).
  - Push-up: `PUSHUP_SAG` (hips dropping, body line under 160°), `PUSHUP_SHALLOW`.
  - Curl: `CURL_SWING` (elbows moving), `CURL_PARTIAL` (not squeezing at the top).
- **Random Forest** (200 trees): takes the summary of one rep and returns a **form score from 0 to 100**.
- The rules say *what* to fix. The model says *how good* the rep was overall.

### Stage 7: Feedback engine *(planned)*
Chooses **one tip at a time** so the user is not overwhelmed.
- Priority: safety (back, knees) before depth before tempo.
- Each tip has a 3-second cooldown.
- After 3 clean reps in a row, it encourages the user.
- Every tip shown is stored for the report.

## 3. The interface (Streamlit) *(planned; pages exist as stubs)*
- **Live page:** webcam view with the skeleton drawn on it, the rep counter, the current tip and the form score. The camera goes through the browser (`streamlit-webrtc`), so it also works inside Docker.
- **Report page:** summary of the finished session (reps, partial reps, average score, most common faults).
- **History page:** progress over time.

## 4. Data storage (SQLite) *(schema done, access code planned)*
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

**Extracted features (done).** `training/extract_keypoints.py` ran MediaPipe over every video (61 minutes, 6 CPU cores). Output: `data/processed/all.parquet` with **445,208 frames from 1,041 videos**, pose detected in 99.9% of frames. Each frame keeps the labels, the 16 features, and the raw 33 landmarks, so new features can be computed later without rerunning MediaPipe.

**Known data issues**
- One file (`subject_024_back_good_diagonal.mp4`) is a corrupt 404-byte download and is excluded.
- Two tricep side-view videos have weak pose detection (52% and 63%). They will be filtered out.

**Visual check (done).** `scripts/render_keypoints.py` draws the skeleton on the videos and writes `data/keypoint_previews/`: 39 annotated clips, an `overview.jpg`, and `quality_report.csv` covering every video. A frame is marked "glitch" when an angle jumps by more than 40° from the previous frame. Results: 220 of 1,041 clips (21%) are flagged, concentrated in specific exercise and view combinations:

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
| Interface | Streamlit, streamlit-webrtc, Plotly |
| Experiment tracking | MLflow (port 5000) |
| Packaging | Docker and docker-compose (app on port 8501), Makefile, Git LFS for model files |

## 8. Project layout

```
ai-workout-coach/
├── app/            Streamlit pages (live, report, history)
├── configs/        app.yaml, rules.yaml, model.yaml
├── data/           raw videos, processed Parquet, labels, SQLite db
├── docker/         Dockerfile
├── docs/           architecture.md
├── models/         pose_landmarker_full.task (+ trained models later)
├── scripts/        render_keypoints.py, eval_rep_counter.py, live_rep_counter.py (done);
│                   benchmark_fps.py, record_clip.py (stubs)
├── src/coach/      capture, pose, features, reasoning, feedback, db, pipeline
├── tests/          unit tests (17 passing)
└── training/       extract_keypoints.py (done) + training scripts (stubs)
```
The slide decks are kept outside this repository.

## 9. Progress

| # | Step | Status |
|---|---|---|
| 1 | Setup: packages, pose model, test runner | ✅ Done |
| 2 | Pose estimator, features, smoothing, window, tests | ✅ Done |
| 3 | Offline extraction of all videos to Parquet | ✅ Done |
| 3b | Visual check of the detections: annotated clips, overview image, quality report for all videos | ✅ Done |
| 4 | Rep counter: glitch filter, thresholds tuned on real curves, self-calibrating state machine, live demo script | ✅ Done |
| 5 | **Form analyzer:** rules checked against the good/bad labels | ⏭ Next |
| 6 | Train the LSTM classifier and Random Forest, split by subject, MLflow tracking | Planned |
| 7 | Runtime: feedback engine, pipeline, database access, Streamlit pages | Planned |
| 8 | Validate on the user's own webcam clips | Planned |

**First usable milestone:** the end of step 4, a webcam rep counter with no ML.

## 10. Risks and open questions

- **Small number of subjects (26).** The model may look good on the dataset and fail on a new person. Mitigation: subject-wise splits, and testing on the user's own clips (step 8).
- **Dataset vs. real webcam.** The clips are controlled recordings; a home webcam has different lighting, framing and angle.
- **Good/bad labels are binary.** The dataset does not say *what* was wrong, so specific tips depend on the hand-written rules.
- **Camera view.** Angles are less reliable from the front and diagonal views, and most reliable from the side. Squat knee-angle minimums confirm it: median 59° (side), 92° (front), 83° (diagonal). The `shoulder_width` feature gives the models a hint about the view. Thresholds should be tuned on the side view first.
- **"Pose detected" does not mean "pose correct".** 99.9% of frames had a detection, but some detections are wrong, mostly on low-resolution clips. Quality filters (visibility, resolution) are needed before training.
- **Real-time speed.** The 20+ FPS target has not been measured yet (`scripts/benchmark_fps.py` is a stub).
