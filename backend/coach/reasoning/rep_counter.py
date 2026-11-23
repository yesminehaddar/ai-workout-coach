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
Seuils par exercice : backend/coach/configs/rules.yaml (rep_counter).

Calibration du haut : certaines personnes (ou angles de caméra) n'atteignent jamais top_above
(ex. bras qui plafonne à 145° alors que le seuil est 150°). On suit donc l'angle maximum des
calib_window_s dernières secondes ; le seuil effectif devient min(top_above, max - top_margin),
sans descendre sous bottom_below + min_range.

Seuils relatifs (ajoutés le 15 nov. 2026) : l'angle 2D dépend de la vue. De biais ou selon
la morphologie, une pompe complète peut ne descendre qu'à 125° et un squat profond à 120°
(seuils fixes à 100° -> toutes les reps « partial »). Quand l'amplitude récente (max - min
des calib_window_s dernières secondes) dépasse min_amplitude, le seuil du bas devient le milieu
de cette amplitude : min + bottom_frac x (max - min), et pas plus bas que bottom_below sauf si
la vue écrase tout l'angle (bras de biais : 72-110° pour une pompe complète). Le haut devient
max - max(top_margin, 15 % de l'amplitude), sans plancher fixe dans ce cas.
Une rep peu profonde est alors comptée, et c'est le FormAnalyzer qui dit « descends plus ».
"""
import math
from collections import deque
from enum import Enum

import numpy as np


class RepState(Enum):
    WAITING = "waiting"      # pas encore vu la position de départ
    TOP = "top"
    GOING_DOWN = "going_down"
    BOTTOM = "bottom"
    GOING_UP = "going_up"


class RepCounter:
    def __init__(self, exercise: str, thresholds: dict, hysteresis: float = 5.0,
                 min_partial_depth: float = 20.0, fps: float = 30.0,
                 calib_window_s: float = 5.0, top_margin: float = 10.0, min_range: float = 30.0,
                 min_amplitude: float = 25.0, bottom_frac: float = 0.5, adaptive_bottom: bool = True):
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
        self.min_amplitude = min_amplitude
        self.bottom_frac = bottom_frac
        self.adaptive_bottom = adaptive_bottom
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
        self.effective_bottom = self.bottom_below
        self._live_bottom = self.bottom_below

    def _calibrate(self, t: float, a: float) -> None:
        self._recent.append((t, a))
        while self._recent and self._recent[0][0] < t - self.calib_window_s:
            self._recent.popleft()
        # Percentiles plutôt que min / max : une seule frame aberrante (erreur de détection)
        # ne doit pas décaler les seuils pendant calib_window_s secondes.
        vals = np.fromiter((v for _, v in self._recent), dtype=float)
        lo, hi = (float(x) for x in np.percentile(vals, [5, 95])) if len(vals) >= 20 else (vals.min(), vals.max())
        amp = hi - lo
        if self.adaptive_bottom and amp >= self.min_amplitude:
            # Mouvement net : seuils relatifs à l'amplitude de la personne (indépendants de la vue).
            top = min(self.top_above, hi - max(self.top_margin, 0.15 * amp))
            mid = lo + self.bottom_frac * amp
            bottom = max(self.bottom_below, mid)
            if bottom > top - 2 * self.hysteresis:  # vue très déformée : on oublie le seuil fixe
                bottom = min(mid, top - 2 * self.hysteresis)
        else:
            top = max(self.bottom_below + self.min_range, min(self.top_above, hi - self.top_margin))
            bottom = self.bottom_below
        self._live_bottom = bottom  # seuil actuel, même en pleine rep (jugement a posteriori)
        # Les seuils ne bougent pas pendant une rep (sinon le bas « suit » la descente).
        if self.state in (RepState.WAITING, RepState.TOP):
            self.effective_top, self.effective_bottom = top, bottom

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
            if a < self.effective_bottom:
                self.state = RepState.BOTTOM
            elif a > top:
                self.state = RepState.TOP
                low = self._mins.get(self.angle_key, top)
                # Au démarrage les seuils ne sont pas encore calibrés : on rejuge la profondeur
                # avec la calibration actuelle, qui inclut maintenant cette rep.
                if low < self._live_bottom:
                    return self._finish(t, counted=True)
                depth = top - low
                if depth >= self.min_partial_depth:
                    return self._finish(t, counted=False)
        elif s == RepState.BOTTOM:
            if a > self.effective_bottom + self.hysteresis:
                self.state = RepState.GOING_UP
        elif s == RepState.GOING_UP:
            if a > top:
                self.state = RepState.TOP
                return self._finish(t, counted=True)
            if a < self.effective_bottom:
                self.state = RepState.BOTTOM  # redescend : même rep
        return None
