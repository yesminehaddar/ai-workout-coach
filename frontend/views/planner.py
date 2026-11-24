"""Chatbot « Gym Bro » : propose une séance de 4-5 exercices (séries, reps, repos) selon le profil
et le groupe musculaire du jour, via l'API Groq (backend/coach/planner/).

Page d'accueil de l'application (lancée par frontend/streamlit_app.py).
"""
import os
import re

import pandas as pd
import streamlit as st

from coach.planner import (
    MUSCLE_GROUPS,
    AthleteProfile,
    GroqClient,
    PlannerError,
    clean_focus,
    generate_plan,
    has_api_key,
    load_env,
)
from ui import configs, profile_repository, setup_page

GREETING = "Hey gym bro, what are we going to hit today?"

setup_page()
st.title("Plan my session")
st.markdown('<p class="muted">Pick a muscle group and Gym Bro builds a 4-5 exercise session '
            'for your profile. Then ask for changes in the chat.</p>', unsafe_allow_html=True)

saved = profile_repository().get()
try:
    profile = AthleteProfile.from_dict(saved) if saved else None
except (ValueError, KeyError):
    profile = None
if profile is None:
    st.info("Fill in your profile first: Gym Bro needs your age, height, weight and goal.")
    st.page_link("views/profile.py", label="Go to Profile", icon=":material/person:")
    st.stop()

load_env()
if not has_api_key():
    st.warning("Gym Bro needs a Groq API key. Create a `.env` file at the project root with the line "
               "`GROQ_API_KEY=your_key` (free key at console.groq.com), then restart the app. "
               "The `.env` file is ignored by git.")
    st.stop()

state = st.session_state


def reset_chat() -> None:
    state.gb_messages = [{"role": "assistant", "text": GREETING}]
    state.gb_focus = None      # groupe musculaire de la séance en cours
    state.gb_plan = None       # dernier plan (dict), renvoyé au modèle pour un ajustement
    state.gb_other_open = False  # champ « other » ouvert


if "gb_messages" not in state:
    reset_chat()


def md(text: str) -> str:
    """Texte venant du modèle ou de l'utilisateur, affiché sans interprétation Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$:])", r"\\\1", text)


def render_plan(plan: dict) -> None:
    st.markdown(f"**{md(plan['title'])}**")
    if plan["message"]:
        st.markdown(md(plan["message"]))
    if plan["warmup"]:
        st.markdown(f"**Warm-up.** {md(plan['warmup'])}")
    columns = {"name": "Exercise", "sets": "Sets", "reps": "Reps", "weight": "Starting weight",
               "rest_s": "Rest (s)", "tip": "Form tip"}
    table = pd.DataFrame(plan["exercises"]).reindex(columns=list(columns)).rename(columns=columns)
    table["Starting weight"] = table["Starting weight"].fillna("-")  # plans d'avant ce champ
    table.index = pd.RangeIndex(1, len(table) + 1)
    st.dataframe(table, width="stretch",
                 column_config={"Form tip": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(table)} exercises · {int(table['Sets'].sum())} working sets · "
               "Weights are starting estimates: pick the load where the last 2 reps of each set are hard "
               "but your form stays clean. Too easy? Add 2.5-5 kg next set. Form breaking? Go lighter.")
    if plan["cooldown"]:
        st.markdown(f"**Cool-down.** {md(plan['cooldown'])}")


def make_client() -> GroqClient:
    cfg = configs()[0].get("planner", {})
    return GroqClient(model=os.environ.get("GROQ_MODEL") or cfg.get("model", "openai/gpt-oss-120b"),
                      temperature=float(cfg.get("temperature", 0.4)),
                      max_tokens=int(cfg.get("max_tokens", 2048)),
                      timeout_s=float(cfg.get("timeout_s", 30)))


def send(user_text: str, focus: str, request: str | None = None) -> None:
    """Affiche le message de l'utilisateur, interroge Gym Bro et garde la réponse dans le chat."""
    state.gb_messages.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(md(user_text))
    try:
        with st.chat_message("assistant"), st.spinner("Gym Bro is building your session..."):
            plan = generate_plan(make_client(), profile, focus, request=request, previous_plan=state.gb_plan)
    except PlannerError as e:
        state.gb_messages.append({"role": "assistant", "text": str(e), "error": True})
    else:
        state.gb_focus, state.gb_plan = focus, plan.to_dict()
        state.gb_messages.append({"role": "assistant", "plan": state.gb_plan})
    st.rerun()


with st.sidebar:
    st.header("Gym Bro")
    st.caption(f"{profile.age} y · {profile.weight_kg:.0f} kg → {profile.target_weight_kg:.0f} kg · "
               f"{profile.goal} · {profile.experience} · {profile.equipment}")
    st.page_link("views/profile.py", label="Edit profile", icon=":material/person:")
    if st.button("New session", width="stretch"):
        reset_chat()
        st.rerun()

for m in state.gb_messages:
    with st.chat_message(m["role"]):
        if m.get("plan"):
            render_plan(m["plan"])
        elif m.get("error"):
            st.error(m["text"])
        else:
            st.markdown(md(m["text"]) if m["role"] == "user" else m["text"])

if state.gb_focus is None:
    cols = st.columns(len(MUSCLE_GROUPS))
    picked = None
    for col, group in zip(cols, MUSCLE_GROUPS):
        if col.button(group.capitalize(), key=f"gb_{group}", width="stretch"):
            picked = group
    if picked == "other":
        state.gb_other_open = True
    elif picked:
        send(f"Let's hit {picked}!", picked)
    if state.gb_other_open:
        with st.form("gb_other_form"):
            other = st.text_input("What do you want to train?", max_chars=40,
                                  placeholder="core, glutes, full body...")
            go = st.form_submit_button("Let's go", type="primary")
        if go:
            try:
                focus = clean_focus("other", other)
            except ValueError as e:
                st.error(str(e).capitalize())
            else:
                state.gb_other_open = False
                send(f"Let's hit {focus}!", focus)
else:
    st.page_link("views/live.py", label="Squats, push-ups or curls in the plan? Let the Live coach check "
                                        "your form", icon=":material/videocam:")
    request = st.chat_input("Want a change? e.g. only dumbbells, swap an exercise, shorter session",
                            max_chars=300)
    if request:
        send(request, state.gb_focus, request=request)
