# Architecture

```
Webcam ─► Camera (OpenCV) ─► PoseEstimator (MediaPipe, 33 pts) ─► Features (angles, filtre, fenêtre 30)
                                                                      │
                     ┌────────────────────────────┬──────────────────┴──────────┐
                     ▼                            ▼                             ▼
          ExerciseClassifier (LSTM)     RepCounter (machine à états)   FormAnalyzer (règles + RF)
                     └────────────────────────────┴──────────────┬──────────────┘
                                                                 ▼
                                               FeedbackEngine (1 conseil) ─► Streamlit
                                                                 │
                                                                 ▼
                                                     SQLite (users → sessions → sets → reps → form_events)
```

| Dossier | Rôle |
|---|---|
| `src/coach/capture` | Stage 1 – webcam |
| `src/coach/pose` | Stage 2 – points du corps |
| `src/coach/features` | Stage 3 – angles, lissage, fenêtre |
| `src/coach/reasoning` | Stages 4-6 – exercice, reps, posture |
| `src/coach/feedback` | Stage 7 – choix du conseil |
| `src/coach/db` | Schéma et accès SQLite |
| `src/coach/pipeline.py` | Boucle par frame qui relie tout |
| `training/` | Pipeline d'entraînement hors-ligne (01→06) |
