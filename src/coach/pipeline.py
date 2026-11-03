"""Orchestrateur : la boucle exécutée sur chaque frame.

Capture -> Pose -> Features -> (Classifier, RepCounter, FormAnalyzer) -> FeedbackEngine -> DB
Budget cible : ~50 ms par frame (20+ FPS sur CPU).
"""
from dataclasses import dataclass


@dataclass
class FrameResult:
    """Ce que l'interface affiche après chaque frame."""
    exercise: str | None
    rep_count: int
    tip: str | None
    form_score: float | None


class CoachPipeline:
    def __init__(self, app_config: dict, rules_config: dict):
        raise NotImplementedError

    def process(self, frame) -> FrameResult:
        """Traite une frame RGB et renvoie l'état courant du coach."""
        raise NotImplementedError
