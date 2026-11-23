"""Lecture à voix haute des consignes (« Go deeper », « Great, keep going »).

À 2 m de l'écran le texte est difficile à lire : la consigne courte de chaque conseil (champ say de
rules.yaml) est dite par la synthèse vocale du système, sur les haut-parleurs du PC qui fait tourner
l'appli. Windows : SAPI (voix installées avec Windows), en choisissant une voix anglaise. Ailleurs :
pyttsx3 s'il est installé, sinon pas de voix (l'appli continue normalement).

Un thread dédié parle ; si une nouvelle consigne arrive pendant qu'il parle, seule la plus récente est
gardée : on ne lit jamais une file de vieilles consignes.
"""
import logging
import queue
import sys
import threading

log = logging.getLogger(__name__)


class Voice:
    def __init__(self, rate: int = 1, language: str = "English"):
        """rate : vitesse SAPI de -10 à 10 (1 = un peu plus vite que la normale)."""
        self.rate, self.language = rate, language
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self.available = False
        self.voice_name = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def say(self, text: str) -> None:
        """Demande de lire text ; remplace une consigne pas encore commencée."""
        if not self.available or not text:
            return
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def close(self) -> None:
        """Arrête le thread (après la consigne en cours)."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(None)

    # ------------------------------------------------------------------ thread
    def _run(self) -> None:
        speak = self._init_engine()
        self._ready.set()
        if speak is None:
            return
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                speak(text)
            except Exception:
                log.warning("voice: could not say %r", text, exc_info=True)

    def _init_engine(self):
        if sys.platform == "win32":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()  # COM dans ce thread
                sapi = win32com.client.Dispatch("SAPI.SpVoice")
                voices = sapi.GetVoices()
                for i in range(voices.Count):
                    if self.language.lower() in voices.Item(i).GetDescription().lower():
                        sapi.Voice = voices.Item(i)
                        break
                sapi.Rate = self.rate
                self.voice_name = sapi.Voice.GetDescription()
                self.available = True
                return lambda text: sapi.Speak(text)
            except Exception:
                log.info("voice: Windows SAPI unavailable", exc_info=True)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self.voice_name, self.available = "pyttsx3", True

            def speak(text: str) -> None:
                engine.say(text)
                engine.runAndWait()
            return speak
        except Exception:  # noqa: BLE001  pas de synthèse vocale : appli sans voix
            log.info("voice: no speech engine, coaching continues without voice")
            return None
