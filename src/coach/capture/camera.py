"""Stage 1 : capture webcam avec OpenCV.

Un thread de capture remplit une file d'1 frame ; le traitement prend toujours
la plus récente, pour que le retard ne s'accumule jamais.
"""


class Camera:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, mirror: bool = True):
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def latest_frame(self):
        """Renvoie la dernière frame en RGB (ou None)."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
