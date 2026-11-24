"""Lecture et vérification du plan renvoyé par le modèle, et génération complète d'un plan.

La réponse du modèle n'est jamais affichée telle quelle : on n'en garde que les champs attendus,
avec des types et des longueurs bornés.
"""
import json
import re
from dataclasses import asdict, dataclass

from coach.planner.groq_client import PlannerError
from coach.planner.profile import AthleteProfile
from coach.planner.prompt import build_messages

MIN_EXERCISES, MAX_EXERCISES = 4, 5


@dataclass(frozen=True)
class Exercise:
    name: str
    sets: int
    reps: str
    rest_s: int
    tip: str
    weight: str = "pick a load: last 2 reps hard"  # charge de départ estimée par le modèle


@dataclass(frozen=True)
class WorkoutPlan:
    title: str
    message: str
    warmup: str
    exercises: tuple[Exercise, ...]
    cooldown: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["exercises"] = [asdict(e) for e in self.exercises]
        return d


class PlanFormatError(PlannerError):
    pass


def _text(value, max_len: int, default: str = "") -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return default
    return " ".join(str(value).split())[:max_len] or default


def _int(value, lo: int, hi: int, default: int) -> int:
    try:
        n = round(float(value))
    except (TypeError, ValueError, OverflowError):  # None, "abc", NaN, Infinity
        return default
    return max(lo, min(hi, n))


def parse_plan(text: str) -> WorkoutPlan:
    """Texte JSON du modèle -> WorkoutPlan. Lève PlanFormatError si c'est inutilisable."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)  # JSON entouré de texte ou de ```
        try:
            data = json.loads(match.group(0)) if match else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("exercises"), list):
        raise PlanFormatError("The coach's answer was not a workout plan.")

    exercises = []
    for item in data["exercises"]:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 80)
        if not name:
            continue
        exercises.append(Exercise(name=name, sets=_int(item.get("sets"), 1, 6, 3),
                                  reps=_text(item.get("reps"), 20, "8-12"),
                                  rest_s=_int(item.get("rest_s"), 15, 300, 60),
                                  tip=_text(item.get("tip"), 160),
                                  weight=_text(item.get("weight"), 40, Exercise.weight)))
    if len(exercises) < MIN_EXERCISES:
        raise PlanFormatError("The coach's plan had fewer than 4 exercises.")
    return WorkoutPlan(title=_text(data.get("title"), 80, "Today's session"),
                       message=_text(data.get("message"), 500),
                       warmup=_text(data.get("warmup"), 300),
                       exercises=tuple(exercises[:MAX_EXERCISES]),
                       cooldown=_text(data.get("cooldown"), 300))


def generate_plan(client, profile: AthleteProfile, focus: str, request: str | None = None,
                  previous_plan: dict | None = None, retries: int = 1) -> WorkoutPlan:
    """Construit le prompt, interroge le modèle (client.chat) et vérifie le plan.

    Une réponse mal formée est redemandée `retries` fois avant d'abandonner.
    """
    messages = build_messages(profile, focus, request=request, previous_plan=previous_plan)
    for attempt in range(retries + 1):
        try:
            return parse_plan(client.chat(messages))
        except PlanFormatError:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")
