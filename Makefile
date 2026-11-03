.PHONY: install run up down test lint data train benchmark

install:      ## Installer les dépendances
	pip install -r requirements-dev.txt

run:          ## Lancer l'application en local
	streamlit run app/streamlit_app.py

up:           ## Lancer l'app + MLflow avec Docker
	docker compose up --build

down:
	docker compose down

test:
	pytest

lint:
	ruff check src app training tests

data:         ## Télécharger les datasets open source
	python scripts/download_datasets.py

train:        ## Entraîner les deux modèles
	python training/train_classifier.py && python training/train_form_model.py

benchmark:    ## Mesurer les FPS sur ce PC
	python scripts/benchmark_fps.py
