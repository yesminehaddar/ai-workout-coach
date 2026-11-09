"""Fenêtre glissante des 30 dernières frames -> tenseur (30, 16) pour le LSTM."""
from collections import deque

import numpy as np

from coach.features.angles import FEATURE_NAMES


class SlidingWindow:
    def __init__(self, size: int = 30):
        self.size = size
        self._buf: deque[list[float]] = deque(maxlen=size)

    def push(self, features: dict[str, float]) -> None:
        self._buf.append([features[k] for k in FEATURE_NAMES])

    def is_full(self) -> bool:
        return len(self._buf) == self.size

    def as_array(self) -> np.ndarray:
        """Tableau (size, n_features). Si la fenêtre n'est pas pleine, complété en tête par la
        première frame (ou des zéros si vide), pour garder une forme fixe."""
        n = len(FEATURE_NAMES)
        if not self._buf:
            return np.zeros((self.size, n), dtype=np.float32)
        arr = np.asarray(self._buf, dtype=np.float32)
        if len(arr) < self.size:
            pad = np.repeat(arr[:1], self.size - len(arr), axis=0)
            arr = np.vstack([pad, arr])
        return arr

    def clear(self) -> None:
        self._buf.clear()
