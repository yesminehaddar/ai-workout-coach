# AI Workout Coach

Coach sportif par webcam : il reconnaît l'exercice (squat, pompe, curl), compte les répétitions
et corrige la posture en direct. Tout tourne en local, en temps réel.

Description complète (concept, pipeline, données, décisions, avancement) : [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).

## Avancement (9 novembre 2026)

| # | Étape | État |
|---|---|---|
| 1 | Installation (paquets, modèle de pose, tests) | ✅ |
| 2 | Détection de pose (MediaPipe) + 16 features + filtre One Euro + fenêtre de 30 frames | ✅ |
| 3 | Extraction de toutes les vidéos vers Parquet (445 208 frames, 1 041 vidéos) + contrôle visuel | ✅ |
| 4 | Compteur de répétitions (88 % des clips à ±1 répétition) + démo webcam | ✅ |
| 5 | Analyse de la posture (règles) | ⏭ Prochaine |
| 6 | Entraînement : LSTM (exercice) + Random Forest (score de forme), séparation par sujet, MLflow | Prévue |
| 7 | Application : moteur de conseils, pipeline, base SQLite, pages Streamlit | Prévue |
| 8 | Validation sur nos propres vidéos webcam | Prévue |

La détection est fiable de profil (curl, squat) mais pas de face pour les pompes : il faut se placer
**de profil** face à la caméra.

## Démarrer

```bash
make install          # pip install -r requirements-dev.txt puis pip install -e .
make test
make live EX=squat    # démo webcam : squat, pushup ou curl  (q = quitter, r = remettre à zéro)
```

Le modèle de pose (`models/pose_landmarker_full.task`, ignoré par Git) se télécharge ici :
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Avec Docker (app Streamlit sur :8501, MLflow sur :5000) :
```bash
cp .env.example .env
docker compose up --build
```

Autres commandes : `make lint`, `make extract` (vidéos → Parquet), `make previews` (squelette dessiné
sur les vidéos + rapport de qualité), `make eval-reps` (évaluation du compteur).

## Données

Dataset Mendeley « fitness exercises, good/bad form » (CC BY 4.0) : 1 042 vidéos, 26 sujets, 7 exercices,
3 vues. Les vidéos et les fichiers extraits ne sont pas dans Git. Voir [data/DATASETS.md](data/DATASETS.md)
et [data/DATASETS_STATUS.md](data/DATASETS_STATUS.md).

## Organisation

```
app/          Interface Streamlit (live, rapport, historique)
configs/      YAML : app, règles et seuils, modèle
data/         raw/ (vidéos), processed/ (Parquet), labels/, db/
docker/       Dockerfile
docs/         architecture.md
models/       Modèle de pose et modèles entraînés
scripts/      Contrôle visuel, évaluation et démo du compteur, benchmark
src/coach/    capture, pose, features, reasoning, feedback, db, pipeline
tests/        Tests unitaires
training/     Extraction → entraînement → évaluation → export
```
