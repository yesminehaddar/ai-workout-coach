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
| `backend/coach/capture` | Stage 1 – webcam |
| `backend/coach/pose` | Stage 2 – points du corps |
| `backend/coach/features` | Stage 3 – angles, lissage, fenêtre |
| `backend/coach/reasoning` | Stages 4-6 – exercice, reps, posture |
| `backend/coach/feedback` | Stage 7 – choix du conseil |
| `backend/coach/db` | Schéma et accès SQLite |
| `backend/coach/pipeline.py` | Boucle par frame qui relie tout |
| `backend/coach/configs` | Seuils, conseils, corrections (YAML) |
| `backend/coach/datasets.py` | Chargement Mendeley + QEVD pour l'entraînement |
| `frontend/streamlit_app.py`, `frontend/views` | Application web : coach en direct, bilan, historique |
| `frontend/ui.py` | Éléments communs des vues (cartes, bilan, bonnes pratiques) |
| `frontend/desktop` | Démo fenêtre OpenCV, vidéo de secours |
| `ml/data` | Lecture des archives, extraction des points, résumés de reps |
| `ml/training` | Entraînement du LSTM et de la Random Forest |
| `ml/evaluation` | Évaluation : compteur, règles, défauts QEVD, FPS |
| `data/models` | Modèle de pose + modèles entraînés (Git LFS) |
| `deploy` | Docker : image de l'appli + MLflow |
