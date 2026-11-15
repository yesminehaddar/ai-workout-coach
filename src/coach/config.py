"""Chargement des fichiers YAML de configs/ et des variables d'environnement (.env)."""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(path: str | Path) -> dict:
    """Lit un fichier YAML et renvoie un dict. Un nom seul ("rules") vise configs/rules.yaml."""
    p = Path(path)
    if not p.suffix:
        p = CONFIG_DIR / f"{p}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
