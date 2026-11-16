.PHONY: install run demo-video up down test lint extract previews eval-reps reps eval-form live train benchmark

install:      ## Installer les dépendances + le package coach (mode éditable)
	pip install -r requirements-dev.txt
	pip install -e .

run:          ## Lancer l'application en local
	python -m streamlit run app/streamlit_app.py

demo-video:   ## Vidéo de démo de secours -> demo/coach_demo.mp4
	python scripts/make_demo_video.py

up:           ## Lancer l'app + MLflow avec Docker
	docker compose up --build

down:
	docker compose down

test:
	pytest

lint:
	ruff check src app training tests scripts

extract:      ## Vidéos -> points + features (data/processed/), reprend là où il s'était arrêté
	python training/extract_keypoints.py --workers 6

previews:     ## Vidéos annotées + rapport de qualité (data/keypoint_previews/)
	python scripts/render_keypoints.py

eval-reps:    ## Évaluer le compteur de répétitions sur le dataset
	python scripts/eval_rep_counter.py

reps:         ## Découper les clips en reps -> data/processed/reps.parquet
	python training/extract_rep_summaries.py

eval-form:    ## Vérifier les règles de posture contre les étiquettes good/bad
	python scripts/eval_form_rules.py

live:         ## Démo webcam du compteur + conseils (EX=squat|pushup|curl)
	python scripts/live_rep_counter.py --exercise $(or $(EX),squat)

train:        ## Entraîner les deux modèles
	python training/train_classifier.py && python training/train_form_model.py

benchmark:    ## Mesurer les FPS sur ce PC
	python scripts/benchmark_fps.py
