"""Filtre One Euro : enlève le tremblement sans ajouter de retard.

Lent -> lissage fort (fréquence de coupure basse) ; rapide -> lissage faible.
min_cutoff : lissage à l'arrêt (Hz, plus petit = plus lisse).
beta : réactivité aux mouvements rapides (plus grand = moins de retard).
"""
import math


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, timestamp: float) -> float:
        """timestamp en secondes, strictement croissant."""
        if self._x is None or self._t is None:
            self._x, self._t = value, timestamp
            return value
        dt = timestamp - self._t
        if dt <= 0:
            return self._x
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (value - self._x) / dt
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * value + (1 - a) * self._x
        self._t = timestamp
        return self._x

    def reset(self) -> None:
        self._x, self._t, self._dx = None, None, 0.0


ANGLE_FEATURES = ("knee", "hip", "elbow", "shoulder", "torso_lean", "body_line")


class JumpFilter:
    """Rejette les erreurs de détection : un angle qui saute de plus de max_jump degrés
    d'une frame à l'autre est remplacé par la dernière valeur acceptée.

    Si le saut persiste plus de max_hold frames, c'est un vrai changement (mouvement rapide,
    personne retrouvée après une perte) : on accepte la nouvelle valeur.
    """

    def __init__(self, max_jump: float = 40.0, max_hold: int = 3, keys=ANGLE_FEATURES):
        self.max_jump = max_jump
        self.max_hold = max_hold
        self.keys = keys
        self._last: dict[str, float] = {}
        self._held: dict[str, int] = {}

    def __call__(self, features: dict[str, float]) -> dict[str, float]:
        out = dict(features)
        for k in self.keys:
            if k not in features:
                continue
            v = features[k]
            last = self._last.get(k)
            if last is not None and abs(v - last) > self.max_jump and self._held.get(k, 0) < self.max_hold:
                self._held[k] = self._held.get(k, 0) + 1
                out[k] = last
            else:
                self._held[k] = 0
                self._last[k] = v
        return out

    def reset(self) -> None:
        self._last.clear()
        self._held.clear()


class FeatureSmoother:
    """Un OneEuroFilter par feature. « side » n'est pas lissé (valeur discrète)."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.05):
        self._kwargs = {"min_cutoff": min_cutoff, "beta": beta}
        self._filters: dict[str, OneEuroFilter] = {}

    def __call__(self, features: dict[str, float], timestamp: float) -> dict[str, float]:
        out = {}
        for k, v in features.items():
            if k == "side":
                out[k] = v
                continue
            f = self._filters.setdefault(k, OneEuroFilter(**self._kwargs))
            out[k] = f(v, timestamp)
        return out

    def reset(self) -> None:
        self._filters.clear()
