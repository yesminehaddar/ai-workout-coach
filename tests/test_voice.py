"""Voix : say() ne bloque jamais la boucle vidéo, même sans synthèse vocale disponible."""
import time

from coach.feedback.voice import Voice


def test_say_is_non_blocking_and_safe_without_engine(monkeypatch):
    monkeypatch.setattr(Voice, "_init_engine", lambda self: None)
    v = Voice()
    assert v.available is False
    t = time.time()
    v.say("Go deeper")  # aucun moteur : ignoré, sans erreur
    assert time.time() - t < 0.1


def test_only_latest_cue_is_kept(monkeypatch):
    spoken = []
    monkeypatch.setattr(Voice, "_init_engine", lambda self: (setattr(self, "available", True), spoken.append)[1])
    v = Voice()
    v.say("Go deeper")
    time.sleep(0.2)
    v.close()
    v._thread.join(timeout=2)
    assert spoken == ["Go deeper"]
