"""Rapport de séance : reps, score de forme par rep, défauts fréquents, lu depuis SQLite."""
import streamlit as st

from coach.ui import exercise_names, render_report, repository, setup_page

setup_page()
st.title("📋 Session report")

sessions = repository().list_sessions()
if not sessions:
    st.info("No session yet. Run one on the **Live coach** page.")
    st.stop()

options = {f"#{s['id']} · {s['started_at']:%d %b %Y, %H:%M} · {exercise_names(s['exercises']) or 'no reps'}": s["id"]
           for s in reversed(sessions)}
choice = st.selectbox("Session", list(options))
render_report(repository().session_report(options[choice]))
