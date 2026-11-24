"""Planificateur de séance (chatbot « Gym Bro ») : profil -> prompt -> Groq -> plan vérifié.

Indépendant du pipeline vidéo : n'importe ni la pose, ni les modèles PyTorch / scikit-learn.
"""
from coach.planner.groq_client import GroqClient, MissingApiKey, PlannerError, has_api_key, load_env
from coach.planner.plan import Exercise, WorkoutPlan, generate_plan, parse_plan
from coach.planner.profile import EQUIPMENT, EXPERIENCE, GENDERS, AthleteProfile
from coach.planner.prompt import MUSCLE_GROUPS, build_messages, clean_focus

__all__ = ["EQUIPMENT", "EXPERIENCE", "GENDERS", "MUSCLE_GROUPS", "AthleteProfile", "Exercise",
           "GroqClient", "MissingApiKey", "PlannerError", "WorkoutPlan", "build_messages", "clean_focus",
           "generate_plan", "has_api_key", "load_env", "parse_plan"]
