"""Stage 7 : choisit UN conseil à la fois.

Priorité : sécurité (dos, genoux) > profondeur > tempo. Cooldown de 3 s par conseil.
Encouragement après 3 reps propres. Chaque conseil est enregistré (form_events).
"""


class FeedbackEngine:
    def __init__(self, cooldown_s: float = 3.0, encourage_after: int = 3):
        raise NotImplementedError

    def choose_tip(self, faults: list, now: float) -> str | None:
        raise NotImplementedError
