"""Stage 7 : choisit UN conseil à la fois.

Priorité : sécurité (dos, genoux) > profondeur > tempo (priority 1 > 2 > 3 dans rules.yaml).
Cooldown de 3 s par conseil : on ne répète pas le même conseil en boucle.
Encouragement après 3 reps propres d'affilée. Chaque conseil donné est gardé dans history.
Si aucune règle ne se déclenche mais que le score du modèle est bas, on donne un conseil
général plutôt que de féliciter (les règles ne voient pas tous les défauts).
"""
from dataclasses import dataclass

from coach.reasoning.form_analyzer import Fault

ENCOURAGEMENTS = ["Great form, keep going!", "Nice and clean!", "Perfect, keep it up!"]
LOW_SCORE = Fault("LOW_SCORE", "Slow down and control the full movement", 3)
LOW_SCORE_BELOW = 50.0


@dataclass
class Tip:
    text: str
    code: str | None  # None = encouragement / message général
    at: float


class FeedbackEngine:
    def __init__(self, cooldown_s: float = 3.0, encourage_after: int = 3):
        self.cooldown_s = cooldown_s
        self.encourage_after = encourage_after
        self.reset()

    def reset(self) -> None:
        self._last_shown: dict[str, float] = {}
        self._clean_streak = 0
        self._n_encourage = 0
        self.history: list[Tip] = []

    def choose_tip(self, faults: list, now: float, score: float | None = None) -> str | None:
        """Appelé après chaque rep avec ses défauts (triés par priorité) et son score.
        Renvoie le conseil à afficher, ou None (rien de nouveau à dire)."""
        if not faults and score is not None and score < LOW_SCORE_BELOW:
            faults = [LOW_SCORE]
        if not faults:
            self._clean_streak += 1
            if self._clean_streak >= self.encourage_after:
                self._clean_streak = 0
                text = ENCOURAGEMENTS[self._n_encourage % len(ENCOURAGEMENTS)]
                self._n_encourage += 1
                self.history.append(Tip(text, None, now))
                return text
            return None
        self._clean_streak = 0
        for f in sorted(faults, key=lambda f: f.priority):
            if now - self._last_shown.get(f.code, float("-inf")) >= self.cooldown_s:
                self._last_shown[f.code] = now
                self.history.append(Tip(f.tip, f.code, now))
                return f.tip
        return None  # tous les défauts sont en cooldown
