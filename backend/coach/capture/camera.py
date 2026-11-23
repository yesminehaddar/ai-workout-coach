"""Stage 1 : capture webcam avec OpenCV (caméra intégrée du portable ou webcam USB branchée).

Un thread de capture garde seulement la frame la plus récente : le traitement prend toujours
la dernière, pour que le retard ne s'accumule jamais quand le pipeline est plus lent que la caméra.

Choix de la caméra : index OpenCV (0 = en général la caméra intégrée, 1, 2... = webcams USB).
list_cameras() teste les premiers index et renvoie ceux qui donnent une image.
Sous Windows on passe par DirectShow : ouverture rapide, et les webcams USB y sont listées.
"""
import sys
import threading
import time
from typing import Self

import cv2
import numpy as np

BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def open_capture(index: int, width: int | None = None, height: int | None = None,
                 fps: int | None = None, fallback: bool = True) -> cv2.VideoCapture:
    """Ouvre la caméra n° index avec la résolution demandée (la caméra peut en choisir une autre)."""
    cap = cv2.VideoCapture(index, BACKEND)
    if fallback and not cap.isOpened() and BACKEND != cv2.CAP_ANY:  # repli sur le backend par défaut
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def list_cameras(max_index: int = 5) -> list[dict]:
    """Caméras qui renvoient une image : [{"index", "width", "height", "label"}, ...]."""
    found, misses = [], 0
    level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # index vides : pas de bruit
    try:
        for i in range(max_index):
            cap = open_capture(i, fallback=False)  # le repli testerait d'autres backends : lent
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if ok and frame is not None:
                misses = 0
                h, w = frame.shape[:2]
                hint = "usually the built-in camera" if i == 0 else "usually a USB / plugged-in camera"
                found.append({"index": i, "width": w, "height": h,
                              "label": f"Camera {i} ({w}x{h}, {hint})"})
            else:
                misses += 1
                if misses >= 2:  # les index sont contigus : après deux trous, plus rien
                    break
    finally:
        cv2.utils.logging.setLogLevel(level)
    return found


class Camera:
    """Webcam lue dans un thread. Utilisation :

        with Camera(index=1) as cam:
            frame, t = cam.latest_frame()
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 mirror: bool = True):
        self.index, self.width, self.height, self.fps, self.mirror = index, width, height, fps, mirror
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        self._cap = open_capture(self.index, self.width, self.height, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera {self.index} could not be opened. Is it connected and not "
                               "used by another application?")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        fails = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails > 50:  # caméra débranchée
                    self._running = False
                time.sleep(0.01)
                continue
            fails = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame, self._t, self._seq = frame, time.time(), self._seq + 1

    @property
    def running(self) -> bool:
        return self._running

    def latest_frame(self, after_seq: int = -1, timeout: float = 2.0):
        """Dernière frame BGR (format OpenCV, comme attendu par CoachPipeline.process).
        Attend une frame plus récente que after_seq. Renvoie (frame, t, seq) ou (None, 0, seq)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None and self._seq > after_seq:
                    return self._frame, self._t, self._seq
            if not self._running:
                break
            time.sleep(0.002)
        return None, 0.0, after_seq

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
