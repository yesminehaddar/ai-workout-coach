"""Stage 6 : analyse de la posture, hybride règles + ML.

- Règles (backend/coach/configs/rules.yaml, form_rules) : chaque défaut -> un conseil explicable.
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

# Vue de la caméra, d'après la largeur des épaules rapportée au torse (moyenne sur la rep) :
# de face ~0.8 (QEVD, 5e percentile 0.5), de profil les épaules se superposent (proche de 0).
# Diagonale : entre les deux ; à recaler sur les vues étiquetées de Mendeley.
FRONT_MIN_SHOULDER_WIDTH = 0.55
VIEWS = ("any", "front", "side")

# Une rep est « bonne » si aucune règle ne se déclenche et que le score atteint ce seuil
# (même seuil que le conseil général du FeedbackEngine).
GOOD_MIN_SCORE = 50.0


@dataclass(frozen=True)
class Fault:
    code: str
    tip: str          # consigne courte pendant la série
    priority: int
    fix: str = ""     # comment corriger : la bonne pratique, en une ou deux phrases
    say: str = ""     # consigne lue à voix haute, 2-3 mots (vide : on lit tip)


@dataclass(frozen=True)
class Rule:
    code: str
    metric: str
    op: str
    value: float
    tip: str
    priority: int
    fix: str = ""
    say: str = ""
    view: str = "any"  # vue où la règle est valable : "front", "side" ou "any"

    @classmethod
    def parse(cls, spec: dict) -> "Rule":
        parts = spec["check"].split()
        if len(parts) != 3 or parts[1] not in OPS:
            raise ValueError(f"Règle {spec.get('code')} : check invalide {spec['check']!r} "
                             "(attendu '<mesure> <op> <valeur>')")
        view = spec.get("view", "any")
        if view not in VIEWS:
            raise ValueError(f"Règle {spec.get('code')} : view invalide {view!r} (attendu {VIEWS})")
        return cls(spec["code"], parts[0], parts[1], float(parts[2]), spec["tip"],
                   int(spec["priority"]), spec.get("fix", ""), spec.get("say", ""), view)

    def fires(self, metrics: dict[str, float]) -> bool:
        v = metrics.get(self.metric)
        if v is None or math.isnan(v):  # mesure absente : on ne juge pas
            return False
        return OPS[self.op](v, self.value)

    def applies(self, metrics: dict[str, float], view: str) -> bool:
        """La règle se déclenche sur cette rep, vue de cette caméra."""
        return self.view in ("any", view) and self.fires(metrics)


def rep_view(metrics: dict[str, float]) -> str:
    """"front" ou "side" d'après la largeur des épaules pendant la rep (inconnue : "side",
    la vue conseillée dans l'appli)."""
    lo, hi = metrics.get("min_shoulder_width"), metrics.get("max_shoulder_width")
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "side"
    return "front" if (lo + hi) / 2 >= FRONT_MIN_SHOULDER_WIDTH else "side"


def verdict(faults: list, score: float | None) -> str:
    """Décision bonne / mauvaise forme pour une rep : "good" ou "bad"."""
    ok = not faults and (score is None or score >= GOOD_MIN_SCORE)
    return "good" if ok else "bad"


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
        self.fixes = {r.code: r.fix for rules in self.rules.values() for r in rules}
        self.model = None
        if model_path:
            import joblib
            self.model = joblib.load(model_path)

    def check_rules(self, exercise: str, rep_summary: dict) -> list[Fault]:
        """Défauts de la rep, triés par priorité (le plus important d'abord)."""
        return self.check_metrics(exercise, rep_metrics(rep_summary))

    def check_metrics(self, exercise: str, metrics: dict[str, float]) -> list[Fault]:
        """Comme check_rules, sur des mesures déjà aplaties (min_X, max_X, range_X, tempo_s)."""
        view = rep_view(metrics)
        faults = [Fault(r.code, r.tip, r.priority, r.fix, r.say)
                  for r in self.rules.get(exercise, []) if r.applies(metrics, view)]
        return sorted(faults, key=lambda f: f.priority)

    def form_score(self, rep_summary: dict, faults: list[Fault] | None = None) -> float:
        """Score 0-100. Avec le modèle (ml/training/train_form_model.py) : probabilité « good » x 100.
        Sinon : 100 moins une pénalité par défaut (30 sécurité, 20 amplitude, 10 tempo / placement)."""
        entry = (self.model or {}).get(rep_summary.get("exercise"))
        if entry is not None:
            m = rep_metrics(rep_summary)
            x = [[0.0 if math.isnan(v := m.get(c, math.nan)) else v for c in entry["columns"]]]
            return float(100.0 * entry["model"].predict_proba(x)[0][1])
        if faults is None:
            faults = self.check_rules(rep_summary["exercise"], rep_summary)
        return max(0.0, 100.0 - sum(PENALTY.get(f.priority, 10.0) for f in faults))
