"""Stage 3 : points -> angles articulaires.

Tout est calculé en 2D (x, y de l'image) : la profondeur z de MediaPipe est trop bruitée.
Les distances sont normalisées par la longueur du torse (indépendant de l'éloignement).
On utilise le côté du corps (gauche ou droit) le mieux visible.
"""
import math

import numpy as np

from coach.pose.estimator import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    SIDE_JOINTS,
)

# Ordre fixe des colonnes (utilisé par la fenêtre, l'extraction et les modèles).
FEATURE_NAMES = [
    "knee",            # angle hanche-genou-cheville
    "hip",             # angle épaule-hanche-genou
    "elbow",           # angle épaule-coude-poignet
    "shoulder",        # angle hanche-épaule-coude (le bras s'écarte du torse)
    "torso_lean",      # inclinaison du torse par rapport à la verticale (0 = droit)
    "body_line",       # angle épaule-hanche-cheville (180 = corps aligné, pompe)
    "wrist_height",    # hauteur du poignet au-dessus de l'épaule / torse
    "hip_height",      # hauteur de la hanche au-dessus de la cheville / torse
    "knee_forward",    # avance du genou devant la cheville / torse (signé vers l'avant)
    "elbow_height",    # hauteur du coude au-dessus de l'épaule / torse
    "shoulder_width",  # largeur des épaules / torse (indique l'angle de vue)
    "knee_spread",     # écart des genoux / écart des chevilles (genoux qui rentrent < 1)
    "elbow_flare",     # écart des coudes / largeur d'épaules
    "wrist_forward",   # poignet devant l'épaule / torse (signé vers l'avant)
    "visibility",      # visibilité moyenne du côté choisi
    "side",            # 0 = gauche, 1 = droit
]


def _xy(p) -> np.ndarray:
    if hasattr(p, "x"):
        return np.array([p.x, p.y], dtype=float)
    return np.asarray(p[:2], dtype=float)


def joint_angle(a, b, c) -> float:
    """Angle en degrés au point b, formé par a-b-c. a, b, c : (x, y) ou objets avec .x/.y."""
    a, b, c = (_xy(p) for p in (a, b, c))
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _pick_side(landmarks) -> str:
    score = {
        side: float(np.mean([landmarks[i].visibility for i in joints]))
        for side, joints in SIDE_JOINTS.items()
    }
    return "left" if score["left"] >= score["right"] else "right"


def compute_features(landmarks, aspect: float = 1.0) -> dict[str, float]:
    """Renvoie toutes les features d'une frame (16 valeurs, voir FEATURE_NAMES).

    aspect = largeur / hauteur de l'image : x est multiplié par aspect pour que les angles
    ne soient pas déformés par les coordonnées normalisées.
    """
    side = _pick_side(landmarks)

    def pt(i):
        p = landmarks[i]
        return np.array([p.x * aspect, p.y], dtype=float)

    if side == "left":
        sh, el, wr, hip, kn, an = (pt(i) for i in (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE))
        o_sh, o_kn, o_an, o_el = pt(R_SHOULDER), pt(R_KNEE), pt(R_ANKLE), pt(R_ELBOW)
    else:
        sh, el, wr, hip, kn, an = (pt(i) for i in (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE))
        o_sh, o_kn, o_an, o_el = pt(L_SHOULDER), pt(L_KNEE), pt(L_ANKLE), pt(L_ELBOW)

    torso = max(float(np.linalg.norm(sh - hip)), 1e-6)

    # Direction « avant » : du côté du nez par rapport à la hanche (le corps regarde vers là).
    forward = 1.0 if landmarks[0].x * aspect >= hip[0] else -1.0

    up = np.array([0.0, -1.0])
    cos = float(np.dot(sh - hip, up)) / torso
    torso_lean = math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))

    sh_width = float(np.linalg.norm(sh - o_sh))
    ankle_gap = float(np.linalg.norm(an - o_an))
    knee_gap = float(np.linalg.norm(kn - o_kn))
    elbow_gap = float(np.linalg.norm(el - o_el))

    return {
        "knee": joint_angle(hip, kn, an),
        "hip": joint_angle(sh, hip, kn),
        "elbow": joint_angle(sh, el, wr),
        "shoulder": joint_angle(hip, sh, el),
        "torso_lean": torso_lean,
        "body_line": joint_angle(sh, hip, an),
        "wrist_height": float((sh[1] - wr[1]) / torso),
        "hip_height": float((an[1] - hip[1]) / torso),
        "knee_forward": float(forward * (kn[0] - an[0]) / torso),
        "elbow_height": float((sh[1] - el[1]) / torso),
        "shoulder_width": sh_width / torso,
        "knee_spread": knee_gap / ankle_gap if ankle_gap > 1e-6 else 1.0,
        "elbow_flare": elbow_gap / sh_width if sh_width > 1e-6 else 1.0,
        "wrist_forward": float(forward * (wr[0] - sh[0]) / torso),
        "visibility": float(np.mean([landmarks[i].visibility for i in SIDE_JOINTS[side]])),
        "side": 0.0 if side == "left" else 1.0,
    }
