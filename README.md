# AI Workout Coach: Real-Time Exercise Coaching from a Webcam

A personal trainer that runs on a laptop with a webcam. It recognizes the exercise (squat, push-up, bicep curl), counts the repetitions, judges the posture on each rep and gives one short tip at a time. Everything runs locally and in real time: no video leaves the computer.

> **Status (17 November 2026):** steps 1 to 7 of 8 are done. The Streamlit app runs the full coach on a demo video, an uploaded video or the webcam: it recognizes the exercise, counts reps, gives one tip at a time with a form score, and saves every session. Validation on our own webcam clips (step 8) is next.
>
> **Run the demo:** `python -m streamlit run app/streamlit_app.py`. See [DEMO.md](DEMO.md) for the presentation guide.

## 1. What This Project Achieved

1. Built a pose pipeline on MediaPipe Pose Landmarker: 33 body landmarks per frame, turned into 16 scale-independent features.
2. Smoothed the landmarks with a One Euro filter and kept a 30-frame sliding window, ready for a sequence classifier.
3. Extracted the features of the whole dataset into Parquet: 445,208 frames from 1,041 videos, pose detected in 99.9% of frames.
4. Audited detection quality on every video with annotated clips and a quality report, and dropped the camera views that were unreliable.
5. Built a rep counter (state machine, no ML) whose thresholds adapt to the user and camera view: 79% of 296 clips exact and 92% within ±1 rep of an automatic peak count (tuned on the same clips, so optimistic), 32 to 41 FPS live on a CPU.
6. Built a rule-based form analyzer: 11 rules tuned on the good/bad labels, 72% of 244 clips classified correctly, with a tip and a score per rep in the live demo.
7. Trained the LSTM exercise classifier: 96.7% of 2-second windows and 98% of clips right on 5 subjects never seen in training (3 min on the GPU), and the Random Forest form score: 74% of clips right on held-out subjects (curl 82%, squat 71%, push-up 68%).
8. Built the application: a pipeline that runs every stage on each frame (25 to 32 FPS on a CPU), a feedback engine (one tip at a time, safety first, 3-second cooldown), a SQLite session history and a three-page Streamlit interface (live coach, session report, history).

## 2. Verified Results

| Measure | Result |
| --- | --- |
| Videos processed | 1,041 (one corrupt file excluded) |
| Frames extracted | 445,208, pose detected in 99.9% |
| Extraction time | 61 minutes on 6 CPU cores |
| Rep counter, all clips (296, side and diagonal views; tuned on these clips) | 79% exact, 92% within ±1 rep |
| Rep counter, side view (within ±1) | curl 100%, squat 96%, push-up 88% |
| Rep counter, diagonal view (within ±1) | curl 100%, push-up 90%, squat 78% |
| Form rules, clips classified good/bad (244; tuned on these clips) | 72%: curl 76%, push-up 71%, squat 65% |
| Exercise classifier (LSTM), 5 unseen subjects | 96.7% of windows, 98% of clips |
| Form score (Random Forest), held-out subjects | 74% of clips (curl 82%, squat 71%, push-up 68%) |
| Live speed, rep counter alone | 32 to 41 FPS on the CPU |
| Live speed, full pipeline (pose, LSTM, counter, form) | 25 to 32 FPS on the CPU |
| Unit tests | 34 passing |

**How to read these.** Only the LSTM and Random Forest rows are measured on people the models never saw. The LSTM row is a single split with 5 test people, not a cross-validation. The rep counter and the form rules were tuned on the clips they are scored on, so they are optimistic. The dataset has no hand-counted reps: the counter is compared with an offline peak count of the same angle curve, which is itself sometimes wrong. The demo clips were hand-picked and are not evidence of typical accuracy.

## 3. Current State

| # | Step | Status |
| --- | --- | --- |
| 1 | Setup: packages, pose model, test runner | ✅ Done |
| 2 | Pose estimator, 16 features, One Euro smoothing, 30-frame window, tests | ✅ Done |
| 3 | Offline extraction of all videos to Parquet | ✅ Done |
| 3b | Visual check of the detections: annotated clips, overview image, quality report | ✅ Done |
| 4 | Rep counter: glitch filter, thresholds tuned on real curves, state machine with adaptive thresholds, live demo | ✅ Done |
| 5 | Form analyzer: 11 rules tuned and checked against the good and bad labels, tips and score in the live demo | ✅ Done |
| 6 | Train the LSTM classifier and the Random Forest, split by subject, MLflow tracking | ✅ Done |
| 7 | Runtime: feedback engine, pipeline, database access, Streamlit pages, backup demo video | ✅ Done |
| 8 | Validate on our own webcam clips | ⏭ Next |

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
- A One Euro filter removes jitter without adding lag, and a 30-frame window (2 seconds, sampled at 15 frames per second) feeds the classifier as a `(30, 16)` array.

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
- Both thresholds adapt to the user and the camera view: a rep counts once the angle drops past the midpoint of the person's own recent range (fixed thresholds missed most diagonal-view reps). Shallow reps are counted and flagged by the form analyzer. Earlier figures (88% within ±1) counted partial reps as reps and overstated accuracy; the current figures count only the reps shown to the user.

| Exercise | Angle | Top (above) | Bottom (below) | Typical lowest angle in the data |
| --- | --- | --- | --- | --- |
| Squat | knee | 160° | 100° | 66 to 72° |
| Push-up | elbow | 150° | 100° | 72 to 79° |
| Curl | elbow | 145° | 80° | 57 to 65° |

Thresholds were set from the real angle curves and live in `configs/rules.yaml`.

### Phase E: Form analyzer (rules done)

- Every clip is cut into reps (4,620 reps, `data/processed/reps.parquet`). Good and bad clips of the same person were compared to find what differs: push-up hips sag and hands too far forward, curl reps partial and rushed; squat faults vary by person.
- 11 rules in `configs/rules.yaml` (`"<measure> <op> <value>"`), each firing on only 5-9% of good reps. A clip is judged bad when over 30% of its reps have a fault: 72% of 244 clips correct (curl 76%, push-up 71%, squat 65%), on the clips used to tune them.
- A Random Forest (200 trees, one per exercise, `training/train_form_model.py`) scores each rep from 0 to 100. Cross-validated by subject: 74% of clips right. It replaces the temporary rule-based score. The rules say what to fix, the model says how good the rep was overall.

### Phase F: Exercise classifier, feedback and app (done)

- PyTorch LSTM (2 layers x 64 units, `training/train_classifier.py`) on the 30-frame window: `squat`, `pushup`, `curl` or `idle` (the 4 other dataset exercises stand in for idle). The label changes only after 5 consecutive votes above 70% confidence. Exported as TorchScript. On 5 unseen subjects: 96.7% of windows (squat 93%, curl 97%, push-up 100%) and 98% of clips.
- Feedback engine: one tip at a time, safety before depth before tempo, 3-second cooldown, encouragement after 3 clean reps.
- `coach/pipeline.py` runs every stage on each frame. In auto mode, a rep counter runs in parallel for each exercise, so the reps done while the LSTM is still deciding (about 3 s) are not lost.
- If no rule fires but the Random Forest score is low, the coach gives a general tip ("Slow down and control the full movement") instead of praise.
- Streamlit interface (`app/`): live coach (demo video, uploaded video or webcam), session report and history. Sessions are saved in SQLite (users, sessions, sets, reps, form events) with SQLAlchemy.
- `scripts/make_demo_video.py` renders a backup demo video (`demo/coach_demo.mp4`, 3 min): six clips, good and bad form for each exercise, with the full overlay.

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
| `training/extract_rep_summaries.py` | Cut every clip into reps, one summary row per rep | Done |
| `scripts/eval_form_rules.py` | Check the posture rules against the good/bad labels | Done |
| `scripts/live_rep_counter.py` | Webcam demo: skeleton, count and state | Done |
| `app/streamlit_app.py` | The application: live coach, session report, history | Done |
| `scripts/make_demo_video.py` | Backup demo video with the full overlay | Done |
| `training/train_classifier.py`, `training/train_form_model.py` | Train the LSTM and the Random Forest | Done |
| `scripts/qevd_split_zip.py`, `training/extract_qevd.py`, `scripts/eval_qevd_squat.py` | Read, extract and test the QEVD dataset (front-view squats) | Done |
| `scripts/benchmark_fps.py`, `scripts/record_clip.py` | Speed benchmark, record own clips | Planned |

## 8. Reproducible Run Order

**1. Install and test.**

```bash
make install          # pip install -r requirements-dev.txt, then pip install -e .
make test
```

**2. Get the pose model.** Download [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) into `models/` (it is not in Git).

**3. Run the app.**

```bash
make run              # = python -m streamlit run app/streamlit_app.py, then open http://localhost:8501
make demo-video       # backup video: demo/coach_demo.mp4
```

The trained models (`models/exercise_lstm.pt`, `models/form_model.joblib`) are stored with [Git LFS](https://git-lfs.com): run `git lfs install` before cloning, or `git lfs pull` after. Retraining takes about 4 minutes with `training/train_classifier.py` and `training/train_form_model.py`, but first needs the dataset and the extraction of step 4 (about an hour). The demo videos come from that dataset and are not in Git; without it, use an uploaded video or the webcam. Stand side-on to the camera: detection is reliable from the side, not from the front.

**4. Rebuild the dataset results** (needs the videos in `data/raw/`, see below).

```bash
make extract          # videos to Parquet
make previews         # annotated clips and quality report
make eval-reps        # rep counter evaluation
make reps             # cut clips into reps
make eval-form        # posture rules evaluation
```

**With Docker** (Streamlit app on port 8501, MLflow on port 5000; the webcam mode needs the app to run locally, not in Docker):

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
├── scripts/        quality checks, evaluations, demo video, QEVD tools
├── src/coach/      capture, pose, features, reasoning, feedback, db, pipeline
├── tests/          unit tests (34 passing)
└── training/       extraction, then training, evaluation and export
```

## 11. Risks and Open Questions

- **26 subjects only.** A model can look good on the dataset and fail on a new person: subject-wise splits and tests on our own clips (step 8).
- **Dataset vs real webcam.** The clips are controlled recordings, a home webcam has different light, framing and angle.
- **Binary labels.** The dataset says good or bad, not what was wrong, so specific tips come from the hand-written rules.
- **Camera view.** Angles are most reliable from the side: the median lowest squat knee angle is 59° from the side, 92° from the front and 83° from the diagonal.
- **Real-time target.** The full pipeline runs at 25 to 32 FPS on a CPU with dataset videos. With the webcam, low light lowers the camera's own frame rate.

## 12. Tech Stack

| Area | Tools |
| --- | --- |
| Vision | OpenCV, MediaPipe Pose Landmarker |
| ML | PyTorch (LSTM), scikit-learn (Random Forest) |
| Data | pandas, Parquet (pyarrow), NumPy |
| Storage | SQLite via SQLAlchemy |
| Interface | Streamlit (webcam through OpenCV) |
| Tooling | MLflow, Docker, docker-compose, Makefile, pytest |

Full project description: [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).

## License

The code is released under the [MIT License](LICENSE).

The datasets and the pose model are not part of this repository and keep their own licenses: the Mendeley
"fitness exercises, good and bad form" dataset is CC BY 4.0 (cite the authors), QEVD-FIT-300k is under
Qualcomm's research-use data license, and the MediaPipe Pose Landmarker model is under Apache 2.0.
