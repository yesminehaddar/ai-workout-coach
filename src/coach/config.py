"""Chargement des fichiers YAML de configs/ et des variables d'environnement (.env)."""
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Lit un fichier YAML et renvoie un dict."""
    raise NotImplementedError
