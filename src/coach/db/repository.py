"""Lectures / écritures utilisées par l'app et le rapport de séance."""
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from coach.db.models import ExerciseSet, FormEvent, Rep, Session, User


class SessionRepository:
    def __init__(self, factory: sessionmaker):
        self.factory = factory

    def get_or_create_user(self, name: str = "Demo") -> int:
        with self.factory() as db:
            user = db.scalar(select(User).where(User.name == name))
            if user is None:
                user = User(name=name)
                db.add(user)
                db.commit()
            return user.id

    def start_session(self, user_id: int) -> int:
        with self.factory() as db:
            s = Session(user_id=user_id, started_at=datetime.now())
            db.add(s)
            db.commit()
            return s.id

    def start_set(self, session_id: int, exercise: str) -> int:
        with self.factory() as db:
            st = ExerciseSet(session_id=session_id, exercise=exercise)
            db.add(st)
            db.commit()
            return st.id

    def save_rep(self, set_id: int, rep_summary: dict, form_score: float | None = None,
                 rep_index: int | None = None) -> int:
        """rep_summary : la sortie de RepCounter.update (une rep terminée)."""
        with self.factory() as db:
            rep = Rep(set_id=set_id, rep_index=rep_index or 0,
                      min_angle=float(rep_summary.get("min_angle") or 0.0),
                      max_angle=float(rep_summary.get("max_angle") or 0.0),
                      tempo_s=float(rep_summary.get("tempo_s") or 0.0),
                      form_score=form_score, counted=bool(rep_summary.get("counted", True)))
            db.add(rep)
            st = db.get(ExerciseSet, set_id)
            if rep.counted:
                st.rep_count += 1
            else:
                st.partial_count += 1
            db.commit()
            return rep.id

    def log_event(self, rep_id: int, fault_code: str, tip: str) -> None:
        with self.factory() as db:
            db.add(FormEvent(rep_id=rep_id, fault_code=fault_code, tip_text=tip, shown_at=datetime.now()))
            db.commit()

    def end_session(self, session_id: int, duration_s: float) -> None:
        with self.factory() as db:
            s = db.get(Session, session_id)
            scores = [r.form_score for st in s.sets for r in st.reps
                      if r.counted and r.form_score is not None]
            s.duration_s = duration_s
            s.avg_score = sum(scores) / len(scores) if scores else None
            db.commit()

    def save_pipeline_session(self, log_reps: list, duration_s: float, user: str = "Demo") -> int:
        """Enregistre d'un coup une séance terminée (liste de RepRecord du pipeline)."""
        sid = self.start_session(self.get_or_create_user(user))
        sets: dict[str, int] = {}
        for r in log_reps:
            if r.exercise not in sets:
                sets[r.exercise] = self.start_set(sid, r.exercise)
            rid = self.save_rep(sets[r.exercise], r.summary, r.score, r.index)
            for f in r.faults:
                self.log_event(rid, f.code, f.tip)
        self.end_session(sid, duration_s)
        return sid

    def session_report(self, session_id: int) -> dict:
        with self.factory() as db:
            s = db.get(Session, session_id)
            sets = []
            for st in s.sets:
                faults: dict[str, int] = {}
                for r in st.reps:
                    for e in r.events:
                        faults[e.fault_code] = faults.get(e.fault_code, 0) + 1
                sets.append({"exercise": st.exercise, "reps": st.rep_count, "partial": st.partial_count,
                             "scores": [r.form_score for r in st.reps if r.counted],
                             "faults": faults})
            return {"id": s.id, "started_at": s.started_at, "duration_s": s.duration_s,
                    "avg_score": s.avg_score, "sets": sets}

    def list_sessions(self) -> list[dict]:
        with self.factory() as db:
            rows = db.scalars(select(Session).order_by(Session.started_at)).all()
            return [{"id": s.id, "started_at": s.started_at, "duration_s": s.duration_s,
                     "avg_score": s.avg_score,
                     "reps": sum(st.rep_count for st in s.sets),
                     "exercises": ", ".join(st.exercise for st in s.sets)} for s in rows]
