"""Stage 2 : MediaPipe PoseLandmarker (BlazePose) -> 33 landmarks (x, y, z, visibility).

x et y sont normalisés (0-1) dans l'image ; z est relatif aux hanches (peu fiable, on ne s'en sert pas).
Le modèle .task est téléchargé dans models/ (voir README).
"""
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "pose_landmarker_full.task"

# Indices MediaPipe utilisés par le projet.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Pour chaque côté : les articulations nécessaires pour juger un mouvement.
SIDE_JOINTS = {
    "left": (L_SHOULDER, L_ELBOW, L_HIP, L_KNEE, L_ANKLE),
    "right": (R_SHOULDER, R_ELBOW, R_HIP, R_KNEE, R_ANKLE),
}


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_visibility: float = 0.5,
        model_path: str | Path | None = None,
        frame_interval_ms: int = 33,
    ):
        # model_complexity est gardé pour la compatibilité avec app.yaml ; on choisit le modèle
        # par son chemin (seul « full » est téléchargé pour l'instant).
        self.model_complexity = model_complexity
        self.min_visibility = min_visibility
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"Modèle de pose introuvable : {path}")
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._interval = frame_interval_ms
        self._next_ts = 0

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int | None = None) -> list[Landmark] | None:
        """Renvoie les 33 points du corps, ou None si personne n'est visible.

        timestamp_ms doit croître d'un appel à l'autre ; sinon il avance de frame_interval_ms.
        """
        if timestamp_ms is None:
            timestamp_ms = self._next_ts
        self._next_ts = max(self._next_ts, timestamp_ms) + self._interval
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None
        return [
            Landmark(p.x, p.y, p.z, getattr(p, "visibility", 1.0) or 0.0)
            for p in result.pose_landmarks[0]
        ]

    def key_joints_visible(self, landmarks: list[Landmark]) -> bool:
        """Faux -> le coach se met en pause : « Step back so I can see you ».

        Il suffit qu'un côté du corps (gauche ou droit) soit entièrement visible.
        """
        return any(
            all(landmarks[i].visibility >= self.min_visibility for i in joints)
            for joints in SIDE_JOINTS.values()
        )

    def close(self) -> None:
        self._landmarker.close()
