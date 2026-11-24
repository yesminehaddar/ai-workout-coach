"""Profil sportif saisi sur la page Profile : validé ici avant d'entrer dans la base ou dans un prompt.

Tous les champs sont des nombres ou des choix dans une liste fermée : aucun texte libre du profil
n'arrive jusqu'au modèle de langage.
"""
from dataclasses import asdict, dataclass

GENDERS = ("male", "female", "prefer not to say")
EXPERIENCE = ("beginner", "intermediate", "advanced")
EQUIPMENT = ("full gym", "dumbbells only", "bodyweight only")

# Bornes de saisie : larges, mais elles écartent les fautes de frappe (178 kg pour 78 kg passe, 1780 non).
LIMITS = {"age": (14, 90), "height_cm": (120.0, 230.0), "weight_kg": (30.0, 250.0),
          "target_weight_kg": (30.0, 250.0)}


@dataclass(frozen=True)
class AthleteProfile:
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    target_weight_kg: float
    experience: str = "beginner"
    equipment: str = "full gym"

    def __post_init__(self) -> None:
        for field, (lo, hi) in LIMITS.items():
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
                raise ValueError(f"{field.replace('_', ' ')} must be between {lo:g} and {hi:g}")
        for field, allowed in (("gender", GENDERS), ("experience", EXPERIENCE), ("equipment", EQUIPMENT)):
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field} must be one of: {', '.join(allowed)}")

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100) ** 2

    @property
    def weight_delta_kg(self) -> float:
        """Négatif : perdre du poids ; positif : en prendre."""
        return self.target_weight_kg - self.weight_kg

    @property
    def goal(self) -> str:
        """Objectif déduit du poids cible (±1 kg = maintien)."""
        if self.weight_delta_kg <= -1:
            return "lose fat"
        if self.weight_delta_kg >= 1:
            return "build muscle"
        return "maintain and recompose"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AthleteProfile":
        """Construit depuis un dict (base ou formulaire) en ignorant les clés inconnues."""
        return cls(age=int(data["age"]), gender=data["gender"], height_cm=float(data["height_cm"]),
                   weight_kg=float(data["weight_kg"]), target_weight_kg=float(data["target_weight_kg"]),
                   experience=data.get("experience", "beginner"),
                   equipment=data.get("equipment", "full gym"))
