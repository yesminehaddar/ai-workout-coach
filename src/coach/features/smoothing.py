"""Filtre One Euro : enlève le tremblement sans ajouter de retard."""


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0):
        raise NotImplementedError

    def __call__(self, value: float, timestamp: float) -> float:
        raise NotImplementedError
