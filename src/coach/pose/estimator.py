"""Stage 2 : MediaPipe Pose (BlazePose) -> 33 landmarks (x, y, z, visibility)."""
from dataclasses import dataclass


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    def __init__(self, model_complexity: int = 1, min_visibility: float = 0.5):
        raise NotImplementedError

    def detect(self, frame_rgb) -> list[Landmark] | None:
        """Renvoie les 33 points du corps, ou None si personne n'est visible."""
        raise NotImplementedError

    def key_joints_visible(self, landmarks: list[Landmark]) -> bool:
        """Faux -> le coach se met en pause : « Step back so I can see you »."""
        raise NotImplementedError
