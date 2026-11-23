"""Suivi MLflow commun aux scripts d'entraînement : tout reste dans data/mlruns/ (non versionné).

MLflow range par défaut les fichiers des runs (copies des modèles) dans ./mlruns, relatif au dossier
de lancement : on fixe donc explicitement la base (mlflow.db) et le dossier des artefacts.
"""
from pathlib import Path

import mlflow

MLRUNS = Path(__file__).resolve().parents[2] / "data" / "mlruns"


def start_experiment(name: str) -> None:
    """Active l'expérience name, en la créant avec ses artefacts dans data/mlruns/artifacts."""
    MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(MLRUNS / "artifacts" / name).as_uri())
    mlflow.set_experiment(name)
