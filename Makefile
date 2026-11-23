.PHONY: install run demo-video up down test lint extract previews eval-reps reps eval-form live train benchmark

install:      ## Installer les dépendances + le package coach (mode éditable)
	pip install -e ".[dev]"

run:          ## Lancer l'application en local
	python -m streamlit run frontend/streamlit_app.py

demo-video:   ## Vidéo de démo de secours -> data/demo/coach_demo.mp4
	python frontend/desktop/make_demo_video.py

up:           ## Lancer l'app + MLflow avec Docker
	docker compose -f deploy/docker-compose.yml up --build

down:
	docker compose -f deploy/docker-compose.yml down

test:
	pytest

lint:
	ruff check backend frontend ml tests

extract:      ## Vidéos -> points + features (data/processed/), reprend là où il s'était arrêté
	python ml/data/extract_keypoints.py --workers 6

previews:     ## Vidéos annotées + rapport de qualité (data/keypoint_previews/)
	python ml/data/render_keypoints.py

eval-reps:    ## Évaluer le compteur de répétitions sur le dataset
	python ml/evaluation/eval_rep_counter.py

reps:         ## Découper les clips en reps -> data/processed/reps.parquet
	python ml/data/extract_rep_summaries.py

eval-form:    ## Vérifier les règles de posture contre les étiquettes good/bad
	python ml/evaluation/eval_form_rules.py

live:         ## Démo webcam du compteur + conseils (EX=squat|pushup|curl)
	python frontend/desktop/live_rep_counter.py --exercise $(or $(EX),squat)

train:        ## Entraîner les deux modèles
	python ml/training/train_classifier.py && python ml/training/train_form_model.py

benchmark:    ## Mesurer les FPS sur ce PC
	python ml/evaluation/benchmark_fps.py
