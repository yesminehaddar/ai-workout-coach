# Demo Guide

How to present the AI Workout Coach in about 10 minutes, with or without the webcam.

## Before the class (5 minutes)

1. Open a terminal in `ai-workout-coach/` and start the app:

   ```bash
   python -m streamlit run app/streamlit_app.py
   ```

   The browser opens on http://localhost:8501. If it does not, open that address yourself.
2. Run one demo clip once (any of them): the first run loads the models and takes a few extra seconds.
3. Keep `demo/coach_demo.mp4` ready in a video player as plan B.
4. For the webcam: good light (a dark room makes the camera slow), the whole body in view, standing **side-on**.

## The demo, step by step

**1. The idea (1 min).** A personal trainer on a laptop with a webcam. It recognizes the exercise, counts the reps, judges the posture of each rep and gives one tip at a time. Everything runs locally; no video leaves the computer.

**2. Good form (2 min).** Sidebar: *Demo video* → **Squat · good form**, exercise **Auto-detect**, press **▶ Start**.
- After about 3 seconds the exercise card switches from "Detecting…" to **Squat**: the LSTM recognized it.
- The rep counter goes up with each squat; scores stay around 90+.
- After 3 clean reps the coach encourages: "Great form, keep going!".
- At the end, the **session report** appears: 16 reps, average form about 95.

**3. Bad form (2 min).** **Push-up · bad form** (the same person as the good push-up clip).
- The coach says **"Tighten your core, hips up"**: the hips sag.
- Scores drop to near 0. Report: 8 reps, the main issue is "Hips sagging".
- Optional: **Squat · bad form** ("Go a bit lower", "Slow down") or **Bicep curl · bad form** ("Squeeze to the top").

**4. History (30 s).** Left menu → **History**: every session is saved in SQLite, with the average score over time. **Session report** reopens any past session.

**5. Webcam (optional, 2 min).** Sidebar: *Webcam* → **▶ Start**. Do a few squats side-on. Press **■ Stop** to end the session and see the report.

## How it works (for questions)

| Stage | What it does | Tool |
|---|---|---|
| 1. Pose | Finds 33 body points in each frame | MediaPipe Pose Landmarker |
| 2. Features | 16 numbers per frame: joint angles, positions, ratios; smoothed | One Euro filter, 30-frame window |
| 3. Exercise recognition | squat / push-up / curl / other, from 2 s of movement | PyTorch LSTM |
| 4. Rep counting | Down-then-up cycles of the knee or elbow angle; adapts to the person and camera view | State machine, no ML |
| 5. Form check | Named mistakes and a 0–100 score per rep | 11 rules + Random Forest |
| 6. Feedback | One tip at a time, safety first, 3-second cooldown | Feedback engine |

## Key numbers

Measured on people the models never saw during training:

- **Exercise recognition:** 96.7% of 2-second windows, 98% of clips.
- **Form score:** 74% of clips classified good/bad correctly (curl 82%, squat 71%, push-up 68%).

Measured on the clips used for tuning, so optimistic:

- **Rep counter:** 79% of clips exact, 92% within ±1 rep (296 clips; side view: curl 100%, squat 96%, push-up 88%), against an automatic count, not a hand count.
- **Form rules:** 72% of clips.

Other:

- **Speed:** 25–32 frames per second on a laptop CPU, the whole pipeline.
- **Data:** 1,041 videos of 26 people (Mendeley dataset), 445,000 frames processed.

## Honest limits (good to say before being asked)

- **The demo clips are hand-picked**: chosen because the rep count matched and the good/bad contrast was clear, and they were used in training. They show how the app works, not how accurate it is.
- **The exercise recognition figure** comes from a single split with 5 test people.
- **26 people only**, filmed in gyms. A home webcam is different; validating on our own clips is step 8.
- **Front view:** push-ups and curls are not reliable from the front (the arms hide the body). The app asks for a side view.
- **The dataset only says "good" or "bad"**, not which mistake. The specific tips come from rules tuned on the data.
- **Next:** the QEVD dataset (Qualcomm, about 1,900 people at home) labels specific squat mistakes. On a first sample of 362 clips (cross-validation AUC): narrow stance 0.95, back not straight 0.92, shallow 0.88, knees over toes 0.75. On the official test split, with only 13 to 31 positive clips per mistake: 0.96, 0.97, 0.88 and 0.68. QEVD has no person IDs, so the same person may be on both sides. The full download is in progress.

## Plan B

- **The app does not start:** play `demo/coach_demo.mp4` (3 min: all six clips with the overlay, then a summary).
- **The webcam is not detected:** use the demo videos; they show exactly the same pipeline.
- **"Could not open the video source":** close other apps using the camera (Teams, Zoom), then press Start again.
- **Start over cleanly:** stop the terminal (Ctrl+C) and run the start command again. To empty the history, delete `data/db/coach.db`.
