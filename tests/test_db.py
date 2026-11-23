"""Base SQLite en mémoire : création du schéma, enregistrement d'une séance, rapport."""
from coach.db.database import init_db
from coach.db.repository import SessionRepository
from coach.reasoning.form_analyzer import Fault


class FakeRep:
    def __init__(self, exercise, index, counted, score, faults):
        self.exercise, self.index, self.counted, self.score, self.faults = exercise, index, counted, score, faults
        self.summary = {"counted": counted, "min_angle": 70.0, "max_angle": 170.0, "tempo_s": 2.0}


def test_save_and_report_session():
    repo = SessionRepository(init_db("sqlite:///:memory:"))
    reps = [FakeRep("squat", 1, True, 90.0, []),
            FakeRep("squat", 2, True, 70.0, [Fault("SQUAT_SHALLOW", "Go a bit lower", 2)]),
            FakeRep("squat", None, False, 40.0, [])]
    sid = repo.save_pipeline_session(reps, duration_s=30.0)
    rep = repo.session_report(sid)
    assert rep["avg_score"] == 80.0
    assert rep["sets"][0]["reps"] == 2 and rep["sets"][0]["partial"] == 1
    assert rep["sets"][0]["faults"] == {"SQUAT_SHALLOW": 1}
    assert rep["sets"][0]["verdicts"] == ["good", "bad"]  # reps comptées seulement
    assert repo.list_sessions()[0]["reps"] == 2
