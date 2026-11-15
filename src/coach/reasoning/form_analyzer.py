"""Stage 6 : analyse de la posture, hybride règles + ML.

- Règles (configs/rules.yaml, form_rules) : chaque défaut -> un conseil explicable.
  Une règle s'écrit "<mesure> <op> <valeur>", par ex. "min_knee > 85". Les mesures viennent du
  résumé de rep du RepCounter : min_X / max_X / range_X (X = une feature) et tempo_s.
- Random Forest (scikit-learn, étape 6) sur le vecteur de la rep -> score de forme 0-100.
  Tant qu'aucun modèle n'est fourni, le score vient des règles (pénalité par défaut).
"""
import math
import operator
from dataclasses import dataclass

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}

# Points retirés au score par défaut, selon la priorité (1 = sécurité).
PENALTY = {1: 30.0, 2: 20.0, 3: 10.0}


@dataclass(frozen=True)
class Fault:
    code: str
    tip: str
    priority: int


@dataclass(frozen=True)
class Rule:
    code: str
    metric: str
    op: str
    value: float
    tip: str
    priority: int

    @classmethod
    def parse(cls, spec: dict) -> "Rule":
        parts = spec["check"].split()
        if len(parts) != 3 or parts[1] not in OPS:
            raise ValueError(f"Règle {spec.get('code')} : check invalide {spec['check']!r} "
                             "(attendu '<mesure> <op> <valeur>')")
        return cls(spec["code"], parts[0], parts[1], float(parts[2]), spec["tip"],
                   int(spec["priority"]))

    def fires(self, metrics: dict[str, float]) -> bool:
        v = metrics.get(self.metric)
        if v is None or math.isnan(v):  # mesure absente : on ne juge pas
            return False
        return OPS[self.op](v, self.value)


def rep_metrics(rep_summary: dict) -> dict[str, float]:
    """Aplatit un résumé de rep (sortie de RepCounter.update) en mesures pour les règles."""
    mins, maxs = rep_summary.get("features_min", {}), rep_summary.get("features_max", {})
    m = {"tempo_s": float(rep_summary.get("tempo_s", math.nan))}
    for k in mins.keys() & maxs.keys():
        m[f"min_{k}"], m[f"max_{k}"] = mins[k], maxs[k]
        m[f"range_{k}"] = maxs[k] - mins[k]
    return m


class FormAnalyzer:
    def __init__(self, rules: dict, model_path: str | None = None):
        """rules : le contenu de rules.yaml (ou directement sa section form_rules)."""
        form_rules = rules.get("form_rules", rules)
        self.rules = {ex: [Rule.parse(r) for r in specs] for ex, specs in form_rules.items()}
        self.model = None
        if model_path:
            import joblib
            self.model = joblib.load(model_path)

    def check_rules(self, exercise: str, rep_summary: dict) -> list[Fault]:
        """Défauts de la rep, triés par priorité (le plus important d'abord)."""
        metrics = rep_metrics(rep_summary)
        faults = [Fault(r.code, r.tip, r.priority)
                  for r in self.rules.get(exercise, []) if r.fires(metrics)]
        return sorted(faults, key=lambda f: f.priority)

    def form_score(self, rep_summary: dict, faults: list[Fault] | None = None) -> float:
        """Score 0-100. Avec le modèle (training/train_form_model.py) : probabilité « good » x 100.
        Sinon : 100 moins une pénalité par défaut (30 sécurité, 20 amplitude, 10 tempo / placement)."""
        entry = (self.model or {}).get(rep_summary.get("exercise"))
        if entry is not None:
            m = rep_metrics(rep_summary)
            x = [[0.0 if math.isnan(v := m.get(c, math.nan)) else v for c in entry["columns"]]]
            return float(100.0 * entry["model"].predict_proba(x)[0][1])
        if faults is None:
            faults = self.check_rules(rep_summary["exercise"], rep_summary)
        return max(0.0, 100.0 - sum(PENALTY.get(f.priority, 10.0) for f in faults))
