"""Stage 6 : analyse de la posture, hybride règles + ML.

- Règles (configs/rules.yaml) : chaque défaut -> un conseil explicable.
- Random Forest (scikit-learn) sur le vecteur de la rep -> score de forme 0-100.
"""
from dataclasses import dataclass


@dataclass
class Fault:
    code: str
    tip: str
    priority: int


class FormAnalyzer:
    def __init__(self, rules: dict, model_path: str | None = None):
        raise NotImplementedError

    def check_rules(self, exercise: str, rep_summary: dict) -> list[Fault]:
        raise NotImplementedError

    def form_score(self, rep_summary: dict) -> float:
        raise NotImplementedError
