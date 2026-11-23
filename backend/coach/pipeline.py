"""Orchestrateur : la boucle exécutée sur chaque frame.

Capture -> Pose -> Features -> (Classifier, RepCounter, FormAnalyzer) -> FeedbackEngine -> DB
Budget cible : ~50 ms par frame (20+ FPS sur CPU).

Mode automatique : le LSTM reconnaît l'exercice. Pour ne pas perdre les reps faites pendant
qu'il se décide (~3 s), un compteur tourne en parallèle pour chaque exercice ; quand le label
devient X, on garde les reps de X terminées dans les REP_GRACE_S dernières secondes.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from coach.features.angles import FEATURE_NAMES, compute_features
from coach.features.smoothing import FeatureSmoother, JumpFilter
from coach.feedback.engine import FeedbackEngine
from coach.pose.estimator import PoseEstimator
from coach.pose.skeleton import draw_skeleton, put_text
from coach.reasoning.classifier import ExerciseClassifier
from coach.reasoning.form_analyzer import FormAnalyzer, verdict
from coach.reasoning.rep_counter import RepCounter

ROOT = Path(__file__).resolve().parents[2]
EXERCISES = ("squat", "pushup", "curl")
CLASSIFIER_FPS = 15.0  # le LSTM a appris sur des fenêtres de 30 frames à 15 fps (2 s)
REP_GRACE_S = 4.0
TIP_SHOW_S = 3.0
NO_POSE_MSG_S = 1.0
SAY_REPEAT_S = 6.0  # une même consigne n'est pas relue avant ce délai
PARTIAL_TIP, PARTIAL_SAY = "Go a bit further", "Go further"
PARTIAL_FIX = ("That rep did not reach the bottom position, so it was not counted. Move through "
               "the full range of motion on every rep, even if you need fewer reps or less weight.")


@dataclass
class RepRecord:
    exercise: str
    index: int | None
    counted: bool
    t: float
    score: float
    faults: list
    summary: dict
    verdict: str = "bad"  # décision de forme : "good" ou "bad"


@dataclass
class FrameResult:
    """Ce que l'interface affiche après chaque frame."""
    exercise: str | None
    rep_count: int
    tip: str | None
    form_score: float | None
    confidence: float = 0.0
    pose_found: bool = False
    partial_count: int = 0
    new_rep: RepRecord | None = None
    fix: str | None = None       # comment corriger le défaut du conseil affiché
    verdict: str | None = None   # décision de forme de la dernière rep
    say: str | None = None       # consigne à lire à voix haute maintenant (une seule fois)


@dataclass
class SessionLog:
    started_at: float = field(default_factory=time.time)
    reps: list[RepRecord] = field(default_factory=list)
    tips: list = field(default_factory=list)


class CoachPipeline:
    def __init__(self, app_config: dict, rules_config: dict, exercise: str = "auto"):
        """exercise : "auto" (reconnaissance par le LSTM) ou "squat" / "pushup" / "curl"."""
        self.app, self.rules = app_config, rules_config
        self.mode = exercise
        self.est = PoseEstimator(min_visibility=app_config["pose"]["min_visibility"])
        cc = app_config["classifier"]
        self.classifier = None
        if exercise == "auto":
            self.classifier = ExerciseClassifier(str(ROOT / cc["model_path"]), cc["min_confidence"],
                                                 cc["votes_to_switch"])
        fm = ROOT / app_config["form_model"]["model_path"]
        self.analyzer = FormAnalyzer(rules_config, str(fm) if fm.exists() else None)
        fb = app_config["feedback"]
        self.feedback = FeedbackEngine(fb["cooldown_s"], fb["encourage_after_clean_reps"])
        self.reset()

    def reset(self) -> None:
        self.jf, self.sm = JumpFilter(), FeatureSmoother()
        self.counters = {ex: RepCounter(ex, self.rules["rep_counter"][ex]) for ex in EXERCISES}
        self.base = dict.fromkeys(EXERCISES, 0)        # reps comptées avant activation
        self.pending: dict[str, list] = {ex: [] for ex in EXERCISES}  # reps non encore attribuées
        self.window: list[list[float]] = []
        self.last_push = -1.0
        self.n_frames = 0
        self.exercise = None if self.mode == "auto" else self.mode
        self.confidence = 1.0 if self.mode != "auto" else 0.0
        self.tip, self.tip_at, self.fix = None, -1e9, ""
        self._say: str | None = None
        self._said_at: dict[str, float] = {}
        self.last_verdict: str | None = None
        self.last_score: float | None = None
        self.last_pose_t = 0.0
        self.log = SessionLog()
        self.feedback.reset()

    def close(self) -> None:
        self.est.close()

    # ------------------------------------------------------------------ boucle
    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        """Traite une frame BGR (OpenCV) prise à l'instant t (secondes) et renvoie l'état."""
        self.n_frames += 1
        h, w = frame_bgr.shape[:2]
        lm = self.est.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), timestamp_ms=int(t * 1000))
        self._landmarks = lm
        new_rep = None
        if lm is not None and self.est.key_joints_visible(lm):
            self.last_pose_t = t
            f = self.sm(self.jf(compute_features(lm, aspect=w / h)), t)
            self._classify(f, t)
            new_rep = self._count(f, t)
        elif t - self.last_pose_t > NO_POSE_MSG_S:
            self._set_tip("Step back so I can see you", t, log=False, say="Step back",
                          fix="Your whole body must be in the picture. Stand side-on, 2-3 m from the camera.")
        ex = self.exercise
        c = self.counters[ex] if ex else None
        showing = t - self.tip_at < TIP_SHOW_S
        return FrameResult(
            exercise=ex, rep_count=(c.count - self.base[ex]) if c else 0,
            tip=self.tip if showing else None, form_score=self.last_score,
            confidence=self.confidence, pose_found=lm is not None,
            partial_count=c.partial_count if c else 0, new_rep=new_rep,
            fix=(self.fix or None) if showing else None, verdict=self.last_verdict,
            say=self._pop_say())

    def _classify(self, f: dict, t: float) -> None:
        if self.classifier is None:
            return
        if t - self.last_push < 1.0 / CLASSIFIER_FPS - 1e-3:
            return
        self.last_push = t
        self.window.append([f[k] for k in FEATURE_NAMES])
        self.window = self.window[-30:]
        if len(self.window) < 30 or self.n_frames % 5:  # ~3 prédictions par seconde suffisent
            return
        label, conf = self.classifier.predict(np.array(self.window, dtype=np.float32))
        self.confidence = conf
        new = label if label in EXERCISES else None
        if new != self.exercise:
            self.exercise = new
            if new:
                kept = [r for r in self.pending[new] if r.t >= t - REP_GRACE_S]
                self.base[new] = self.counters[new].count - sum(r.counted for r in kept)
                for i, r in enumerate([r for r in kept if r.counted], 1):
                    r.index = i
                self.log.reps.extend(kept)
                self.pending = {ex: [] for ex in EXERCISES}

    def _count(self, f: dict, t: float) -> RepRecord | None:
        active = None
        for ex, c in self.counters.items():
            ev = c.update(f, timestamp=t)
            if ev is None:
                continue
            faults = self.analyzer.check_rules(ex, ev)
            score = self.analyzer.form_score(ev, faults)
            rec = RepRecord(ex, (c.count - self.base[ex]) if ev["counted"] else None, ev["counted"], t,
                            score, faults, ev, verdict(faults, score) if ev["counted"] else "bad")
            if ex == self.exercise:
                self.log.reps.append(rec)
                active = rec
            else:
                self.pending[ex].append(rec)
                self.pending[ex] = [r for r in self.pending[ex] if r.t >= t - REP_GRACE_S]
        if active is not None:
            self.last_score = active.score
            self.last_verdict = active.verdict
            tip = self.feedback.choose_tip(active.faults, t, active.score)
            fix = self.feedback.last_fix if tip else ""
            say = self.feedback.last_say if tip else ""
            if not active.counted and not tip:
                tip, fix, say = PARTIAL_TIP, PARTIAL_FIX, PARTIAL_SAY
            if tip:
                self._set_tip(tip, t, fix=fix, say=say)
        return active

    def _set_tip(self, text: str, t: float, log: bool = True, fix: str = "", say: str = "") -> None:
        if say and t - self._said_at.get(say, -1e9) >= SAY_REPEAT_S:
            self._said_at[say] = t
            self._say = say
        if text != self.tip or t - self.tip_at >= TIP_SHOW_S:
            self.tip, self.tip_at, self.fix = text, t, fix
            if log:
                self.log.tips.append((t, text))

    def _pop_say(self) -> str | None:
        say, self._say = self._say, None
        return say

    # ------------------------------------------------------------------ affichage
    def annotate(self, frame_bgr: np.ndarray, r: FrameResult, panel: bool = True) -> np.ndarray:
        """Dessine le squelette et (si panel) le panneau du coach sur une copie de la frame."""
        img = frame_bgr.copy()
        if self._landmarks is not None:
            draw_skeleton(img, self._landmarks)
        return self.draw_panel(img, r) if panel else img

    def draw_panel(self, img: np.ndarray, r: FrameResult) -> np.ndarray:
        """Bandeau du haut (exercice, reps, score) et conseil en bas, à l'échelle de img."""
        h, w = img.shape[:2]
        s = max(h, w) / 1000
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(120 * s)), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        name = (r.exercise or "...").upper()
        if self.mode == "auto" and r.exercise is None:
            name = "DETECTING..."
        put_text(img, f"{name}   reps: {r.rep_count}", (int(15 * s), int(50 * s)), 1.3 * s, (0, 255, 0))
        sub = f"score {r.form_score:.0f}/100" if r.form_score is not None else "score -"
        sub += f"   partial {r.partial_count}"
        if r.verdict:
            sub += "   GOOD FORM" if r.verdict == "good" else "   FIX FORM"
        color = (255, 255, 255) if r.form_score is None else (
            (113, 204, 46) if r.form_score >= 75 else (18, 156, 243) if r.form_score >= 50 else (60, 76, 231))
        put_text(img, sub, (int(15 * s), int(95 * s)), 0.8 * s, color)
        if r.tip:
            (tw, th), _ = cv2.getTextSize(r.tip, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, 2)
            y = h - int(30 * s)
            cv2.rectangle(img, (0, y - th - int(20 * s)), (w, h), (0, 0, 0), -1)
            put_text(img, r.tip, (max(5, (w - tw) // 2), y), 1.1 * s, (0, 200, 255))
        return img

    # ------------------------------------------------------------------ bilan
    def summary(self) -> dict:
        """Bilan de la séance : par exercice, reps comptées, partielles, score moyen, défauts."""
        out = {}
        for ex in EXERCISES:
            reps = [r for r in self.log.reps if r.exercise == ex]
            if not reps:
                continue
            counted = [r for r in reps if r.counted]
            faults: dict[str, int] = {}
            for r in reps:
                for f in r.faults:
                    faults[f.code] = faults.get(f.code, 0) + 1
            out[ex] = {"reps": len(counted), "partial": len(reps) - len(counted),
                       "good_reps": sum(r.verdict == "good" for r in counted),
                       "avg_score": float(np.mean([r.score for r in counted])) if counted else None,
                       "faults": dict(sorted(faults.items(), key=lambda kv: -kv[1]))}
        return out
