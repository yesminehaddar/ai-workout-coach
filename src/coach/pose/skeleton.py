"""Dessin du squelette MediaPipe sur une image (utilisé par l'appli et les scripts d'aperçu)."""
import cv2
import numpy as np

# Os du corps (sans le visage ni les mains)
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]
LEFT_POINTS = {11, 13, 15, 23, 25, 27, 29, 31}

WHITE = (255, 255, 255)
LEFT_COLOR = (255, 120, 0)    # bleu (BGR)
RIGHT_COLOR = (0, 140, 255)   # orange (BGR)
NOSE_COLOR = (0, 220, 0)


def draw_skeleton(img: np.ndarray, points: dict[int, tuple[float, float]] | list,
                  joints: bool = True) -> np.ndarray:
    """Dessine le squelette sur img (modifiée en place) et la renvoie.

    points : liste de 33 objets avec .x/.y normalisés (sortie de PoseEstimator.detect),
    ou dict {indice: (x, y)} normalisés (ex. lu dans un Parquet).
    """
    h, w = img.shape[:2]

    def px(i):
        p = points[i]
        x, y = (p.x, p.y) if hasattr(p, "x") else p
        return int(x * w), int(y * h)

    thick, radius = max(2, h // 250), max(4, h // 100)
    for a, b in BONES:
        cv2.line(img, px(a), px(b), WHITE, thick)
    if joints:
        for i in range(11, 33):
            cv2.circle(img, px(i), radius, LEFT_COLOR if i in LEFT_POINTS else RIGHT_COLOR, -1)
        cv2.circle(img, px(0), radius, NOSE_COLOR, -1)
    return img


def put_text(img: np.ndarray, text: str, pos: tuple[int, int], scale: float = 1.0,
             color: tuple[int, int, int] = WHITE) -> None:
    """Texte lisible sur n'importe quel fond (contour noir)."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, int(5 * scale)))
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(2 * scale)))
