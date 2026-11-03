"""Stage 5 : machine à états TOP -> GOING_DOWN -> BOTTOM -> GOING_UP -> TOP (= +1 rep).

Demi-tour avant BOTTOM -> répétition « partial », non comptée.
Seuils par exercice : configs/rules.yaml (rep_counter).
"""
from enum import Enum


class RepState(Enum):
    TOP = "top"
    GOING_DOWN = "going_down"
    BOTTOM = "bottom"
    GOING_UP = "going_up"


class RepCounter:
    def __init__(self, exercise: str, thresholds: dict):
        raise NotImplementedError

    def update(self, features: dict[str, float]) -> dict | None:
        """Renvoie un résumé de répétition quand une rep se termine, sinon None."""
        raise NotImplementedError
