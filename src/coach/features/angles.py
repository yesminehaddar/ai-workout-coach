"""Stage 3 : points -> angles articulaires.

knee (hanche-genou-cheville), hip, torso_lean, body_line, elbow, arm_swing.
Normalisation par la longueur du torse ; côté gauche ou droit selon la visibilité.
"""


def joint_angle(a, b, c) -> float:
    """Angle en degrés au point b, formé par a-b-c."""
    raise NotImplementedError


def compute_features(landmarks) -> dict[str, float]:
    """Renvoie toutes les features d'une frame (16 valeurs)."""
    raise NotImplementedError
