"""Caméra : le thread garde la frame la plus récente (caméra simulée, pas de matériel requis)."""
import numpy as np

from coach.capture import camera as cam_mod
from coach.capture.camera import Camera


class FakeCapture:
    def __init__(self, *args):
        self.n = 0

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        self.n += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 0] = 255  # colonne de gauche marquée : doit passer à droite en miroir
        return True, frame

    def release(self):
        pass


def test_latest_frame_is_newer_and_mirrored(monkeypatch):
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCapture)
    with Camera(index=1, mirror=True) as cam:
        f1, _, s1 = cam.latest_frame()
        _, _, s2 = cam.latest_frame(after_seq=s1)
        assert f1.shape == (4, 6, 3) and s2 > s1
        assert f1[0, -1, 0] == 255 and f1[0, 0, 0] == 0


def test_unopenable_camera_raises(monkeypatch):
    class Closed(FakeCapture):
        def isOpened(self):
            return False
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", Closed)
    try:
        Camera(index=7).start()
    except RuntimeError as e:
        assert "Camera 7" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
