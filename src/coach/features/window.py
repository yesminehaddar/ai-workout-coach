"""Fenêtre glissante des 30 dernières frames -> tenseur (30, 16) pour le LSTM."""


class SlidingWindow:
    def __init__(self, size: int = 30):
        raise NotImplementedError

    def push(self, features: dict[str, float]) -> None:
        raise NotImplementedError

    def is_full(self) -> bool:
        raise NotImplementedError

    def as_array(self):
        raise NotImplementedError
