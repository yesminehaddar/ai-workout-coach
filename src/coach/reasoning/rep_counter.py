"""Stage 5 : machine à états TOP -> GOING_DOWN -> BOTTOM -> GOING_UP -> TOP (= +1 rep).

Les trois exercices ont la même forme : l'angle suivi est grand en position de départ
(genou tendu, bras tendu) et petit au milieu du mouvement.
  - TOP        : angle > top_above
  - GOING_DOWN : l'angle est passé sous top_above
  - BOTTOM     : l'angle est passé sous bottom_below
  - GOING_UP   : l'angle remonte au-dessus de bottom_below + hysteresis
  - retour au-dessus de top_above -> une répétition comptée.

Demi-tour avant BOTTOM -> répétition « partial », non comptée (seulement si la descente
dépasse min_partial_depth degrés, pour ignorer le bruit autour du seuil).
On ne compte rien tant qu'on n'a pas vu une première position TOP (départ en plein mouvement).
Seuils par exercice : configs/rules.yaml (rep_counter).

Calibration du haut : certaines personnes (ou angles de caméra) n'atteignent jamais top_above
(ex. bras qui plafonne à 145° alors que le seuil est 150°). On suit donc l'angle maximum des
calib_window_s dernières secondes ; le seuil effectif devient min(top_above, max - top_margin),
sans descendre sous bottom_below + min_range. Le seuil du bas reste fixe : une rep trop peu
profonde reste « partial » (c'est un défaut de forme, signalé par le FormAnalyzer).
"""
import math
from collections import deque
from enum import Enum


class RepState(Enum):
    WAITING = "waiting"      # pas encore vu la position de départ
    TOP = "top"
    GOING_DOWN = "going_down"
    BOTTOM = "bottom"
    GOING_UP = "going_up"


class RepCounter:
    def __init__(self, exercise: str, thresholds: dict, hysteresis: float = 5.0,
                 min_partial_depth: float = 20.0, fps: float = 30.0,
                 calib_window_s: float = 5.0, top_margin: float = 10.0, min_range: float = 30.0):
        self.exercise = exercise
        self.angle_key: str = thresholds["angle"]
        self.top_above: float = float(thresholds["top_above"])
        self.bottom_below: float = float(thresholds["bottom_below"])
        if self.bottom_below >= self.top_above:
            raise ValueError("bottom_below doit être inférieur à top_above")
        self.hysteresis = hysteresis
        self.min_partial_depth = min_partial_depth
        self.fps = fps
        self.calib_window_s = calib_window_s
        self.top_margin = top_margin
        self.min_range = min_range
        self.reset()

    def reset(self) -> None:
        self.state = RepState.WAITING
        self.count = 0
        self.partial_count = 0
        self._frame = 0
        self._start_t: float | None = None
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}
        self._recent: deque[tuple[float, float]] = deque()
        self.effective_top = self.top_above

    def _calibrate(self, t: float, a: float) -> None:
        self._recent.append((t, a))
        while self._recent and self._recent[0][0] < t - self.calib_window_s:
            self._recent.popleft()
        observed = max(v for _, v in self._recent)
        floor = self.bottom_below + self.min_range
        self.effective_top = max(floor, min(self.top_above, observed - self.top_margin))

    def _track(self, features: dict[str, float]) -> None:
        for k, v in features.items():
            if v < self._mins.get(k, float("inf")):
                self._mins[k] = v
            if v > self._maxs.get(k, float("-inf")):
                self._maxs[k] = v

    def _start_rep(self, t: float, features: dict[str, float]) -> None:
        self._start_t = t
        self._mins, self._maxs = {}, {}
        self._track(features)

    def _finish(self, t: float, counted: bool) -> dict:
        if counted:
            self.count += 1
        else:
            self.partial_count += 1
        return {
            "exercise": self.exercise,
            "rep_index": self.count if counted else None,
            "counted": counted,
            "angle": self.angle_key,
            "min_angle": self._mins.get(self.angle_key),
            "max_angle": self._maxs.get(self.angle_key),
            "tempo_s": t - (self._start_t if self._start_t is not None else t),
            "features_min": dict(self._mins),
            "features_max": dict(self._maxs),
        }

    def update(self, features: dict[str, float], timestamp: float | None = None) -> dict | None:
        """Renvoie un résumé de répétition quand une rep se termine (comptée ou partielle),
        sinon None. timestamp en secondes ; par défaut frame / fps."""
        t = timestamp if timestamp is not None else self._frame / self.fps
        self._frame += 1
        a = features.get(self.angle_key)
        if a is None or math.isnan(a):  # pas de pose : on ne change rien
            return None

        self._calibrate(t, a)
        top = self.effective_top
        s = self.state
        if s in (RepState.GOING_DOWN, RepState.BOTTOM, RepState.GOING_UP):
            self._track(features)

        if s == RepState.WAITING:
            if a > top:
                self.state = RepState.TOP
        elif s == RepState.TOP:
            if a < top - self.hysteresis:
                self.state = RepState.GOING_DOWN
                self._start_rep(t, features)
        elif s == RepState.GOING_DOWN:
            if a < self.bottom_below:
                self.state = RepState.BOTTOM
            elif a > top:
                self.state = RepState.TOP
                depth = top - self._mins.get(self.angle_key, top)
                if depth >= self.min_partial_depth:
                    return self._finish(t, counted=False)
        elif s == RepState.BOTTOM:
            if a > self.bottom_below + self.hysteresis:
                self.state = RepState.GOING_UP
        elif s == RepState.GOING_UP:
            if a > top:
                self.state = RepState.TOP
                return self._finish(t, counted=True)
            if a < self.bottom_below:
                self.state = RepState.BOTTOM  # redescend : même rep
        return None
