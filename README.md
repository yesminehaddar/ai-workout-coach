# AI Workout Coach: Real-Time Exercise Coaching from a Webcam

A personal trainer that runs on a laptop with a webcam. It recognizes the exercise (squat, push-up, bicep curl), counts the repetitions, judges the posture on each rep and gives one short tip at a time. Everything runs locally and in real time: no video leaves the computer.

> **Status (9 November 2026):** steps 1 to 4 of 8 are done. The webcam rep counter works today, without machine learning. Posture analysis and the trained models are next. Anything marked *planned* is designed but not yet coded.

## 1. What This Project Achieved

1. Built a pose pipeline on MediaPipe Pose Landmarker: 33 body landmarks per frame, turned into 16 scale-independent features.
2. Smoothed the landmarks with a One Euro filter and kept a 30-frame sliding window, ready for a sequence classifier.
3. Extracted the features of the whole dataset into Parquet: 445,208 frames from 1,041 videos, pose detected in 99.9% of frames.
4. Audited detection quality on every video with annotated clips and a quality report, and dropped the camera views that were unreliable.
5. Built a self-calibrating rep counter (state machine, no ML): 88% of 296 clips within ±1 rep of an independent count, 32 to 41 FPS live on a CPU.
6. Designed the full application: LSTM exercise classifier, hybrid form analyzer, feedback engine, SQLite history and a Streamlit interface.

## 2. Verified Results

| Measure | Result |
| --- | --- |
| Videos processed | 1,041 (one corrupt file excluded) |
| Frames extracted | 445,208, pose detected in 99.9% |
| Extraction time | 61 minutes on 6 CPU cores |
| Rep counter, all clips (296, side and diagonal views) | 88% within ±1 rep |
| Rep counter, side view | curl 100%, squat 88%, push-up 75% |
| Live demo speed | 32 to 41 FPS on the CPU |
| Unit tests | 17 passing |

The dataset has no hand-counted reps, so the counter is compared with an independent offline peak count of each angle curve.

## 3. Current State

| # | Step | Status |
| --- | --- | --- |
| 1 | Setup: packages, pose model, test runner | ✅ Done |
| 2 | Pose estimator, 16 features, One Euro smoothing, 30-frame window, tests | ✅ Done |
| 3 | Offline extraction of all videos to Parquet | ✅ Done |
| 3b | Visual check of the detections: annotated clips, overview image, quality report | ✅ Done |
| 4 | Rep counter: glitch filter, thresholds tuned on real curves, self-calibrating state machine, live demo | ✅ Done |
| 5 | Form analyzer: rules checked against the good and bad labels | ⏭ Next |
| 6 | Train the LSTM classifier and the Random Forest, split by subject, MLflow tracking | Planned |
| 7 | Runtime: feedback engine, pipeline, database access, Streamlit pages | Planned |
| 8 | Validate on our own webcam clips | Planned |

## 4. How It Works

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

## 5. Pipeline Phases (What Was Done)

### Phase A: Pose estimation (done)

Goal: find the body in every frame, reliably.

- MediaPipe PoseLandmarker in video mode, which tracks the person across frames and is smoother than detecting each frame alone.
- Only x and y are used: the depth value z is too noisy.
- If no full side of the body is visible, the coach pauses and asks the user to step back.

### Phase B: Features (done)

Goal: turn 33 landmarks into numbers that describe the movement.

| Group | Features |
| --- | --- |
| Joint angles | knee, hip, elbow, shoulder |
| Posture | torso lean, body line (180 = straight, for push-ups) |
| Positions (normalized) | wrist height, hip height, elbow height, knee forward, wrist forward |
| Shape ratios | shoulder width (hints at the camera view), knee spread (knees caving in), elbow flare |
| Bookkeeping | visibility, side used |

- Distances are divided by torso length, so the result does not depend on how far the user stands.
- The better-visible side of the body is used.
- A One Euro filter removes jitter without adding lag, and a 30-frame window (about 1 second) feeds the classifier as a `(30, 16)` array.

### Phase C: Dataset extraction and quality audit (done)

Goal: build a clean training set and know where detection fails.

- `training/extract_keypoints.py` ran MediaPipe over every video into `data/processed/all.parquet`, keeping the labels, the 16 features and the raw 33 landmarks, so new features can be computed without rerunning MediaPipe.
- `scripts/render_keypoints.py` drew the skeleton on 39 clips and wrote a quality report for all 1,041 videos. A frame is a "glitch" when an angle jumps by more than 40°.

| Exercise | Flagged, front | Flagged, side | Flagged, diagonal |
| --- | --- | --- | --- |
| Push-up | 98% | 6% | 6% |
| Bicep curl | 52% | 0% | 6% |
| Squat | 16% | 8% | 4% |

Decision: front-view push-up and curl clips are dropped, thresholds are tuned on the side view, and the live app asks the user to stand side-on to the camera.

### Phase D: Rep counter (done)

Goal: count reps with no machine learning.

- A state machine follows `TOP → GOING_DOWN → BOTTOM → GOING_UP → TOP`, and returning to TOP adds one rep. Turning back before BOTTOM is a partial rep and is not counted.
- A glitch filter discards angle jumps above 40° unless they last more than 3 frames.
- The top threshold calibrates itself to the user, while the bottom threshold stays fixed so shallow reps are still flagged.

| Exercise | Angle | Top (above) | Bottom (below) | Typical lowest angle in the data |
| --- | --- | --- | --- | --- |
| Squat | knee | 160° | 100° | 66 to 72° |
| Push-up | elbow | 150° | 100° | 72 to 79° |
| Curl | elbow | 145° | 80° | 57 to 65° |

Thresholds were set from the real angle curves and live in `configs/rules.yaml`.

### Phase E: Form analyzer (next)

- Rules, each with a fault code, a tip and a priority: `SQUAT_SHALLOW`, `SQUAT_LEAN`, `PUSHUP_SAG`, `PUSHUP_SHALLOW`, `CURL_SWING`, `CURL_PARTIAL`.
- A Random Forest (200 trees) that scores each rep from 0 to 100. The rules say what to fix, the model says how good the rep was overall.

### Phase F: Exercise classifier, feedback and app (planned)

- PyTorch LSTM (2 layers x 64 units) on the 30-frame window: `squat`, `pushup`, `curl` or `idle`. The label changes only after 5 consecutive votes above 70% confidence. Exported as TorchScript.
- Feedback engine: one tip at a time, safety before depth before tempo, 3-second cooldown, encouragement after 3 clean reps.
- Streamlit interface (live, report, history) through streamlit-webrtc, and a SQLite history (users, sessions, sets, reps, form events) with SQLAlchemy. The schema is done.

## 6. Key Design Decisions

- **Split by person, never by clip.** With only 26 subjects, the same person in training and test would inflate the results.
- **Rep counting without ML.** Angle thresholds are simpler, explainable and already useful on their own.
- **Hybrid form analysis.** Rules give explainable tips, the Random Forest gives an overall score.
- **Same features offline and live.** The same MediaPipe model and feature code run in training and in the app.
- **CPU for pose, GPU for training.** MediaPipe has no GPU support on Windows, PyTorch uses the GPU for the LSTM.
- **Everything configurable in YAML.** Thresholds, tips and model settings change without touching code.

## 7. Script Catalog

| Script | Purpose | Status |
| --- | --- | --- |
| `training/extract_keypoints.py` | Run MediaPipe over every video into Parquet | Done |
| `scripts/render_keypoints.py` | Draw the skeleton on clips, write the quality report | Done |
| `scripts/eval_rep_counter.py` | Evaluate the rep counter against an offline peak count | Done |
| `scripts/live_rep_counter.py` | Webcam demo: skeleton, count and state | Done |
| `scripts/benchmark_fps.py`, `scripts/record_clip.py` | Speed benchmark, record own clips | Planned |

## 8. Reproducible Run Order

**1. Install and test.**

```bash
make install          # pip install -r requirements-dev.txt, then pip install -e .
make test
```

**2. Get the pose model.** Download [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) into `models/` (it is not in Git).

**3. Try the live rep counter.**

```bash
make live EX=squat    # squat, pushup or curl (q to quit, r to reset)
```

Stand side-on to the camera: detection is reliable from the side, not from the front.

**4. Rebuild the dataset results** (needs the videos in `data/raw/`, see below).

```bash
make extract          # videos to Parquet
make previews         # annotated clips and quality report
make eval-reps        # rep counter evaluation
```

**With Docker** (Streamlit app on port 8501, MLflow on port 5000):

```bash
cp .env.example .env
docker compose up --build
```

## 9. Data

Mendeley dataset "fitness exercises, good and bad form" (CC BY 4.0): 1,042 videos, 26 subjects, 7 exercises, each filmed with correct and incorrect form from 3 views (front, side, diagonal). The videos and extracted files are not in Git. See `data/DATASETS.md` and `data/DATASETS_STATUS.md`.

## 10. Project Layout

```
ai-workout-coach/
├── app/            Streamlit pages (live, report, history)
├── configs/        app.yaml, rules.yaml, model.yaml
├── data/           raw videos, processed Parquet, labels, SQLite db (not in Git)
├── docker/         Dockerfile
├── docs/           architecture.md
├── models/         pose model, trained models later
├── scripts/        quality check, rep counter evaluation and live demo
├── src/coach/      capture, pose, features, reasoning, feedback, db, pipeline
├── tests/          unit tests (17 passing)
└── training/       extraction, then training, evaluation and export
```

## 11. Risks and Open Questions

- **26 subjects only.** A model can look good on the dataset and fail on a new person: subject-wise splits and tests on our own clips (step 8).
- **Dataset vs real webcam.** The clips are controlled recordings, a home webcam has different light, framing and angle.
- **Binary labels.** The dataset says good or bad, not what was wrong, so specific tips come from the hand-written rules.
- **Camera view.** Angles are most reliable from the side: the median lowest squat knee angle is 59° from the side, 92° from the front and 83° from the diagonal.
- **Real-time target.** The full pipeline target is 20+ FPS on a CPU. The rep counter demo reaches 32 to 41 FPS, the full pipeline is not measured yet.

## 12. Tech Stack

| Area | Tools |
| --- | --- |
| Vision | OpenCV, MediaPipe Pose Landmarker |
| ML | PyTorch (LSTM), scikit-learn (Random Forest) |
| Data | pandas, Parquet (pyarrow), NumPy |
| Storage | SQLite via SQLAlchemy |
| Interface | Streamlit, streamlit-webrtc, Plotly |
| Tooling | MLflow, Docker, docker-compose, Makefile, pytest |

Full project description: [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).
