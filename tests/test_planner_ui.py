"""Pages Streamlit du chatbot (AppTest) : base temporaire, Groq simulé, aucun appel réseau."""
import json
from pathlib import Path

import pytest

from coach.db.database import init_db
from coach.db.repository import ProfileRepository
from coach.planner import GroqClient

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")
PROFILE = {"age": 30, "gender": "female", "height_cm": 168.0, "weight_kg": 70.0, "target_weight_kg": 65.0,
           "experience": "intermediate", "equipment": "full gym"}
PLAN = json.dumps({"title": "Chest blast", "message": "Let's go!", "warmup": "Band pull-aparts",
                   "exercises": [{"name": f"Press {i}", "sets": 3, "reps": "10-12", "rest_s": 60,
                                  "weight": "2 x 15 kg dumbbells",
                                  "tip": "Shoulders down"} for i in range(5)],
                   "cooldown": "Chest stretch"})


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st

    url = f"sqlite:///{(tmp_path / 'ui.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr("coach.planner.groq_client.ROOT", tmp_path)  # pas de .env réel
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    st.cache_resource.clear()  # repository() lit DATABASE_URL
    yield url
    st.cache_resource.clear()


def run():
    return streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()


def test_planner_is_the_welcome_page_and_asks_for_profile(app):
    at = run()
    assert not at.exception
    assert at.title[0].value == "Plan my session"
    assert "profile first" in at.info[0].value


def test_planner_asks_for_api_key(app):
    ProfileRepository(init_db(app)).save(PROFILE)
    at = run()
    assert not at.exception
    assert "GROQ_API_KEY" in at.warning[0].value


def test_planner_chat_builds_a_plan(app, monkeypatch):
    ProfileRepository(init_db(app)).save(PROFILE)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    sent = []
    monkeypatch.setattr(GroqClient, "chat", lambda self, messages: sent.append(messages) or PLAN)

    at = run()
    assert not at.exception
    assert "what are we going to hit today" in at.chat_message[0].markdown[0].value
    at.button(key="gb_chest").click().run()
    assert not at.exception
    assert "<focus>chest</focus>" in sent[0][1]["content"]
    assert "goal: lose fat" in sent[0][1]["content"]
    table = at.dataframe[0].value
    assert len(table) == 5 and table["Exercise"].iloc[0] == "Press 0"
    assert list(table.columns)[:4] == ["Exercise", "Sets", "Reps", "Starting weight"]
    assert table["Starting weight"].iloc[0] == "2 x 15 kg dumbbells"

    at.chat_input[0].set_value("only dumbbells please").run()
    assert not at.exception
    assert len(sent) == 2 and sent[1][-1]["role"] == "user" and "only dumbbells" in sent[1][-1]["content"]
    assert len(at.dataframe) == 2


def test_planner_other_focus_and_error_in_chat(app, monkeypatch):
    from coach.planner import PlannerError

    ProfileRepository(init_db(app)).save(PROFILE)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def fail(self, messages):
        raise PlannerError("Groq rate limit reached. Wait a minute and try again.")

    monkeypatch.setattr(GroqClient, "chat", fail)
    at = run()
    at.button(key="gb_other").click().run()
    at.text_input[0].input("core <b>& glutes</b>")
    at.button(key="FormSubmitter:gb_other_form-Let's go").click().run()
    assert not at.exception
    assert "rate limit" in at.error[0].value
    said = [m.value for c in at.chat_message for m in c.markdown]
    assert any("glutes" in v for v in said) and not any("<b>" in v for v in said)
    assert len(at.dataframe) == 0 and at.button(key="gb_chest")  # on peut réessayer


def test_old_plan_without_weight_still_renders(app, monkeypatch):
    """Un plan gardé en session avant l'ajout du poids s'affiche avec « - »."""
    ProfileRepository(init_db(app)).save(PROFILE)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    at = run()
    old = json.loads(PLAN)
    for e in old["exercises"]:
        del e["weight"]
    at.session_state["gb_messages"] = [{"role": "assistant", "text": "hi"}, {"role": "assistant", "plan": old}]
    at.session_state["gb_focus"], at.session_state["gb_plan"] = "chest", old
    at.run()
    assert not at.exception
    assert list(at.dataframe[0].value["Starting weight"]) == ["-"] * 5
