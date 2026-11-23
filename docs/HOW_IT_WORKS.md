# How the Workout Coach Works

This document explains, stage by stage, how the coach turns a webcam image into a rep count, a form verdict and a spoken correction. It also covers how the models are trained and evaluated. Every number below comes from the code and the configuration files in this repository as of 24 November 2026.

## 1. The Big Picture

The coach runs a loop on every video frame (about 30 per second), entirely on the local computer:

```
camera frame
   │
   ▼
1 Capture ──► 2 Pose (33 body points) ──► 3 Features (angles, distances) ──► 4 Cleaning (glitch filter, smoothing)
                                                                                   │
                     ┌─────────────────────────────────────────────────────────────┼──────────────────────────┐
                     ▼                                                             ▼                          ▼
          5 Exercise recognition (LSTM)                                 6 Rep counter (state machine)   (rep summary)
                     │                                                             │                          │
                     └──────────────────────────────┬──────────────────────────────┘                          ▼
                                                    ▼                                          7 Form analysis (rules + Random Forest)
                                         8 Feedback: one tip, one spoken cue, one correction ◄─────────────────┘
                                                    │
                                   9 Screen + voice, then 10 session saved to SQLite
```

Stages 1 to 4 run on every frame. The exercise classifier runs about 3 times per second. Form analysis and feedback run once per finished repetition.

The orchestration lives in `backend/coach/pipeline.py` (`CoachPipeline.process`). The web interface (`frontend/`) only calls this loop and displays the result.

## 2. Stage by Stage

### 2.1 Capture

`backend/coach/capture/camera.py`

- **Camera.** The coach can use the built-in laptop camera or a USB webcam. `list_cameras()` tries the first camera indexes and keeps those that return an image; the Live coach page lists them, and **Detect cameras** searches again after a camera is plugged in. On Windows the DirectShow backend is used, which opens quickly and lists USB webcams.
- **No delay build-up.** A background thread reads the camera continuously and keeps only the newest frame. If processing is slower than the camera, old frames are dropped instead of queueing, so the coach always reacts to what is happening now.
- **Settings** (`backend/coach/configs/app.yaml`): 1280×720 requested at 30 frames per second, mirrored like a mirror.
- **Video files** (demo clips and uploads) are read frame by frame at their own frame rate, in real time or as fast as possible.

### 2.2 Pose Estimation

`backend/coach/pose/estimator.py`

- **Model.** MediaPipe Pose Landmarker, "full" version (`data/models/pose_landmarker_full.task`), in video mode: it tracks the person from one frame to the next, which is steadier than detecting each frame independently.
- **Output.** 33 body points per frame (nose, shoulders, elbows, wrists, hips, knees, ankles, feet…), each with x and y as a fraction of the image width and height, plus a visibility score from 0 to 1. The depth value z is too noisy and is not used.
- **When the body is not visible.** Coaching only runs if one complete side of the body (shoulder, elbow, hip, knee and ankle) has a visibility of at least 0.5. After 1 second without that, the coach says "Step back".
- **Speed.** MediaPipe runs on the CPU (it has no GPU support on Windows). The whole pipeline runs at 25 to 32 frames per second on a laptop CPU.

### 2.3 Features: From Points to Measurements

`backend/coach/features/angles.py`

Raw point coordinates depend on where the person stands and how far away they are, so they are turned into measurements that do not:

- **The better-visible side** of the body (left or right) is used for single-side measurements.
- **x is multiplied by the image aspect ratio**, so that angles are not distorted by the normalized coordinates.
- **Distances are divided by the torso length** (shoulder to hip), so the result does not depend on the distance to the camera.

16 features go to the exercise classifier (`FEATURE_NAMES`), and one more is used by the form rules only (`EXTRA_FEATURES`):

| Feature | What it measures | Used for |
| --- | --- | --- |
| `knee` | angle hip-knee-ankle | squat reps and depth |
| `hip` | angle shoulder-hip-knee | squat |
| `elbow` | angle shoulder-elbow-wrist | push-up and curl reps |
| `shoulder` | angle hip-shoulder-elbow | curl: elbow moving forward |
| `torso_lean` | torso angle from vertical (0 = upright) | squat back angle (side), curl swing |
| `body_line` | angle shoulder-hip-ankle (180 = straight) | push-up hips sagging |
| `wrist_height`, `hip_height`, `elbow_height` | heights relative to the torso | squat depth (front), curl |
| `knee_forward` | knee in front of the ankle | squat knees past toes (side) |
| `wrist_forward` | wrist in front of the shoulder | push-up hand placement |
| `shoulder_width` | shoulder width / torso length | camera view; back angle from the front |
| `knee_spread` | knee gap / ankle gap (below 1 = knees caving in) | squat knees caving in (front) |
| `elbow_flare` | elbow gap / shoulder width | push-up |
| `visibility`, `side` | detection quality, side used | bookkeeping |
| `stance_width` (extra) | ankle gap / shoulder width | squat feet too close (front) |

### 2.4 Cleaning the Signal

`backend/coach/features/smoothing.py`

- **Glitch filter** (`JumpFilter`). If one of the six angles jumps by more than 40° from one frame to the next, the jump is treated as a detection error and the previous value is kept, unless the jump lasts more than 3 frames (then it is a real fast movement).
- **One Euro filter** (`FeatureSmoother`, min_cutoff 1.5 Hz, beta 0.05) on every feature. It smooths jitter when the body is still and follows quickly when it moves, so it removes noise without adding noticeable lag.

### 2.5 Exercise Recognition

`backend/coach/reasoning/classifier.py`, trained by `ml/training/train_classifier.py`

- **Input.** A window of the last 30 frames, sampled at 15 frames per second (2 seconds of movement) × 16 features.
- **Network.** PyTorch LSTM, 2 layers of 64 units, dropout 0.3, followed by a linear layer with 4 outputs: `squat`, `pushup`, `curl`, `idle`. The training mean and standard deviation are stored inside the model, so it receives raw features. It is exported as TorchScript (`data/models/exercise_lstm.pt`).
- **Stable decisions.** A prediction is made every 5 frames once the window is full (about 3 per second). The displayed exercise only changes after 5 identical predictions in a row, each with at least 70% confidence. This stops the label from flickering, but it means the coach needs about 3.5 seconds to decide.
- **No lost reps.** While the LSTM is still deciding, a rep counter runs in parallel for each of the 3 exercises. When the label settles on, say, squat, the squat reps finished in the last 4 seconds are kept.
- **Manual mode.** If the user picks the exercise in the sidebar, the LSTM is not used at all.

### 2.6 Rep Counting

`backend/coach/reasoning/rep_counter.py`

Rep counting uses no machine learning. One joint angle is followed per exercise: the knee for squats, the elbow for push-ups and curls. It is large at the start position (leg or arm straight) and small in the middle of the movement.

A state machine follows the angle:

```
WAITING ──(start position seen)──► TOP ──(angle drops below top)──► GOING_DOWN ──(below bottom)──► BOTTOM
   ▲                                                                                                │
   └──────── back above top: +1 rep ◄──────── GOING_UP ◄──────(rises 5° above bottom)───────────────┘
```

- **Starting thresholds** (`rules.yaml`, `rep_counter`):

  | Exercise | Angle | Top (above) | Bottom (below) |
  | --- | --- | --- | --- |
  | Squat | knee | 160° | 100° |
  | Push-up | elbow | 150° | 100° |
  | Curl | elbow | 145° | 80° |

- **Thresholds adapt to the person and the camera.** The 2D angle depends on the viewing angle and the body, so a full push-up seen at an angle may only reach 125°. Over the last 5 seconds, once the movement spans at least 25°, the bottom threshold becomes the middle of the person's own range, and the top becomes the maximum minus 10° or 15% of the range, whichever is larger.
- **Partial reps.** Turning back before reaching the bottom (after at least 20° of descent) is a partial rep: it is not counted, it is shown as "partial", and the coach says "Go further".
- **Rep summary.** When a rep ends, the counter returns the minimum and maximum of every feature during the rep, plus its duration (`tempo_s`). This summary is what the form analysis judges.

### 2.7 Form Analysis

`backend/coach/reasoning/form_analyzer.py`, rules in `backend/coach/configs/rules.yaml`

Form analysis combines explainable rules (what is wrong, how to fix it) with a machine-learning score (how good the rep was overall).

**Camera view.** Some faults can only be seen from one side: knees caving in only from the front, knees past the toes only from the side. Each rep's view is decided from the shoulder width: if the shoulders are at least 0.55 × torso length wide on average, the person faces the camera ("front"), otherwise they stand side-on ("side"). Each rule declares the view it is valid for.

**Rules.** A rule is one measurement compared with one threshold, for example `min_knee > 85` (the knee never bent below 85°). The measurement is `min_`, `max_` or `range_` of a feature during the rep, or `tempo_s`. Each rule has a priority (1 = safety, 2 = range of motion, 3 = tempo or placement), a displayed tip, a spoken cue of 2 or 3 words, and a correction explaining how to fix it.

| Exercise | Fault | Check | View | Spoken cue |
| --- | --- | --- | --- | --- |
| Squat | Leaning forward | `max_torso_lean > 55` | side | Chest up |
| Squat | Back not straight | `max_shoulder_width > 0.96` (leaning toward the camera shortens the torso) | front | Back straight |
| Squat | Knees caving in | `min_knee_spread < 0.80` | front | Knees out |
| Squat | Not deep enough | `min_knee > 85` | side | Go deeper |
| Squat | Not deep enough | `min_hip_height > 1.17` | front | Go deeper |
| Squat | Knees past toes | `max_knee_forward > 0.52` | side | Sit back |
| Squat | Feet too close | `max_stance_width < 0.87` | front | Feet wider |
| Squat | Too fast | `tempo_s < 1.07` | any | Slow down |
| Push-up | Hips sagging | `min_body_line < 148` | side | Hips up |
| Push-up | Not low enough | `min_elbow > 100` | any | Go lower |
| Push-up | Hands too far forward | `max_wrist_forward > -0.02` | any | Hands under shoulders |
| Curl | Swinging the body | `range_torso_lean > 6.5` | any | Don't swing |
| Curl | Elbows moving | `range_shoulder > 19` | any | Pin your elbows |
| Curl | Partial range | `min_elbow > 70` | any | Full range |
| Curl | Too fast | `tempo_s < 1.0` | any | Slow down |

Side-view thresholds were tuned on the Mendeley dataset so that each rule fires on only 5 to 9% of reps labelled good. Front-view thresholds were tuned on QEVD squats (see section 3.4).

**Form score (0 to 100).** A Random Forest (200 trees, one model per exercise, `data/models/form_model.joblib`) reads the rep summary (min, max and range of every feature, plus tempo; visibility and shoulder width are left out so it does not learn the camera) and outputs the probability that the rep is good, × 100. Without the model, the score is 100 minus 30, 20 or 10 points per fault depending on its priority.

**Verdict.** A counted rep has **good form** if no rule fired and the score is at least 50; otherwise it is **form to fix**.

### 2.8 Feedback

`backend/coach/feedback/engine.py`

The engine gives one tip at a time, so the user is not flooded:

- **Priority:** safety first, then range of motion, then tempo and placement. Within the same priority, rules are taken in their order in `rules.yaml`.
- **Cooldown:** the same fault is not repeated within 3 seconds; the next fault in line is given instead.
- **Encouragement:** after 2 clean reps in a row, "Great, keep going" (alternating with "Great, continue" and "Nice, keep it up").
- **Low score without a rule:** if no rule fired but the form score is below 50, the coach gives a general tip ("Slow down and control the full movement"), because the rules do not cover every fault.

Each tip carries three texts: the tip shown on screen ("Go a bit lower"), the short spoken cue ("Go deeper"), and the correction ("Sit down until your thighs are about parallel to the floor…"), which is shown under the tip and in the session report.

### 2.9 Voice

`backend/coach/feedback/voice.py`

At 2 m from the screen the text is hard to read, so the short cue is spoken through the speakers of the computer running the app:

- Windows built-in speech (SAPI), with an English voice chosen automatically (Microsoft Zira on a French Windows). On other systems pyttsx3 is used if installed; otherwise the coach works without voice.
- A dedicated thread speaks, so the video never waits. If a new cue arrives while it is speaking, only the latest is kept.
- The same phrase is not repeated within 6 seconds.
- The "Voice coaching" switch in the sidebar turns it off.

### 2.10 Display and Session History

`frontend/`, `backend/coach/db/`

- **Live coach page** (`frontend/views/live.py`): the video with the skeleton drawn on it, and a dark coach panel showing the exercise, the rep count in large type, the verdict of the last rep, the tip, the correction and the form score.
- **Saving.** When the session stops, it is saved in a local SQLite database (`data/db/coach.db`) through SQLAlchemy: `users` → `sessions` → `sets` (one per exercise) → `reps` (angles, tempo, score, counted or partial) → `form_events` (each fault of each rep).
- **Session report** (`frontend/views/report.py`): total and partial reps, average score, good versus to-fix reps, score per rep, the most frequent faults with their corrections, and a best-practice checklist per exercise.
- **History** (`frontend/views/history.py`): every session and the score trend over time.

## 3. Training and Evaluation (Offline)

Everything in `ml/` runs offline, before the app is used. It produces the two trained models in `data/models/`.

### 3.1 Datasets

| Dataset | Content | Used for |
| --- | --- | --- |
| Mendeley "fitness exercises, good and bad form" (CC BY 4.0) | 1,042 phone videos, 26 people, 7 exercises, each filmed with good and bad form from the front, side and diagonal | the original LSTM and Random Forest, side-view rule thresholds |
| QEVD-FIT-300k (Qualcomm research licence) | short home-webcam clips filmed from the front, with named faults per clip ("shallow", "knees over toes", "back not straight", "narrow"…) | front-view rules, front-view training; squats and push-ups only |

QEVD comes as split archives (`.z01`, `.z02`, `.zip`). `ml/data/qevd_split_zip.py build` reads them without recombining, extracts the readable squat and push-up clips and writes their labels from `fine_grained_labels.json`. With Part 1 complete and only the last piece of Parts 2 to 4, 1,810 clips are readable (981 squats, 829 push-ups).

### 3.2 From Videos to Training Tables

1. **Body points.** `ml/data/extract_keypoints.py` (Mendeley) and `ml/data/extract_qevd.py` (QEVD) run the same MediaPipe model and the same feature code as the app on every frame, and save one row per frame in Parquet: video information, the 16 features and the raw 33 points (`data/processed/all.parquet`, `qevd.parquet`). Keeping the raw points lets new features, such as `stance_width`, be computed later without running MediaPipe again.
2. **Reps.** `ml/data/extract_rep_summaries.py` replays every clip through the same glitch filter, smoothing and rep counter as the app, and writes one row per rep (`data/processed/reps.parquet`) with the clip's label: good or bad, plus the named faults for QEVD.

`backend/coach/datasets.py` loads both datasets the same way and handles their differences.

### 3.3 Training

- **Splits by person, never by clip.** Mendeley has 26 people: 5 are kept for the test and 3 for validation, so a model is always judged on people it never saw. QEVD has no person ID; its own train/test split is kept, 10% of its training clips are used for validation, and its test clips are never used for training. QEVD scores are slightly optimistic, because one person can appear in several clips.
- **Classifier** (`ml/training/train_classifier.py`): windows of 30 frames with a step of 5 frames, Adam, class-weighted cross-entropy, early stopping on validation accuracy. It trains on the GPU if available (about 1 minute on an RTX 3050).
- **Form model** (`ml/training/train_form_model.py`): Random Forest per exercise, 5-fold cross-validation grouped by person (by clip for QEVD).
- Both scripts use every dataset present in `data/processed/`, print results per dataset and camera view, and log runs to MLflow in `data/mlruns/`. `--out data/models/candidates/…` writes new models next to the current ones so they can be compared before replacing them.

### 3.4 Evaluation and Current Results

`ml/evaluation/eval_qevd.py` measures rep counting, rules, form score and exercise recognition on QEVD. The Mendeley evaluations are `eval_rep_counter.py` and `eval_form_rules.py`.

| What | Result | Measured on |
| --- | --- | --- |
| Rep counter | 79% of clips exact, 92% within ±1 rep | 296 Mendeley clips, side and diagonal (thresholds tuned on the same clips) |
| Rep counter | squats 80% exact (97% within ±1), push-ups 74% (93%) | 1,492 QEVD clips, front view |
| Exercise classifier (current) | 96.7% of windows, 98% of clips | 5 Mendeley people never seen |
| Form score (current) | 74% of clips right | Mendeley, cross-validated by person |
| Front-view squat rules | back not straight 54% caught / 0% false alarms; feet too close 91% / 5%; depth 53% / 7% | QEVD test clips |

The dataset has no hand-counted reps, so the counter is compared with an offline count of the dips in each angle curve.

## 4. Where to Change Things

| To change | Edit |
| --- | --- |
| A threshold, a tip, a spoken cue, a correction, or the view of a rule | `backend/coach/configs/rules.yaml` (`form_rules`) |
| Rep-counting angles and thresholds | `backend/coach/configs/rules.yaml` (`rep_counter`) |
| Camera resolution, LSTM confidence and votes, tip cooldown, encouragement after N clean reps | `backend/coach/configs/app.yaml` |
| LSTM size and training settings, number of trees | `backend/coach/configs/model.yaml` |
| Colours and layout of the web app | `.streamlit/config.toml`, `frontend/ui.py` |
| Best-practice checklists in the report | `BEST_PRACTICE` in `frontend/ui.py` |

## 5. Code Map

| Path | Role |
| --- | --- |
| `frontend/streamlit_app.py`, `frontend/views/` | web app: live coach, session report, history |
| `frontend/ui.py` | shared look: theme colours, coach panel, report |
| `frontend/desktop/` | OpenCV window demo, backup demo video |
| `backend/coach/capture/` | camera choice and background capture |
| `backend/coach/pose/` | MediaPipe pose, skeleton drawing |
| `backend/coach/features/` | features, glitch filter, smoothing, window |
| `backend/coach/reasoning/` | exercise classifier, rep counter, form analyzer |
| `backend/coach/feedback/` | tip selection, voice |
| `backend/coach/db/` | SQLite schema and queries |
| `backend/coach/pipeline.py` | the per-frame loop that connects everything |
| `backend/coach/datasets.py` | Mendeley and QEVD loading for training |
| `ml/data/`, `ml/training/`, `ml/evaluation/` | dataset tools, training, evaluation |
| `tests/` | unit tests (`.\tasks.ps1 test`) |

## 6. Known Limits

- **Camera view.** 2D angles change with the viewing angle. Rules exist per view, but the front/side split (shoulder width 0.55) is not yet calibrated on labelled side and diagonal clips, and the current form score model was trained on side views only.
- **Few people.** The original models learned from 26 people; tests on new people and home webcams are still needed.
- **Labels.** Mendeley only says good or bad per clip, not which fault; QEVD names faults but covers squats and push-ups from the front only, and has no person ID.
- **Light and framing.** Low light lowers the camera frame rate, and the whole body must stay in the picture.
