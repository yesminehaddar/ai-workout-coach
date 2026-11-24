"""Point d'entrée de l'application : navigation entre les pages. Le chatbot « Gym Bro » ouvre l'appli.

Lancer : python -m streamlit run frontend/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent  # absolu : marche quel que soit le dossier de lancement
st.set_page_config(page_title="Workout Coach", page_icon=":material/fitness_center:", layout="wide")
nav = st.navigation([
    st.Page(HERE / "views" / "planner.py", title="Plan my session", default=True),
    st.Page(HERE / "views" / "live.py", title="Live coach"),
    st.Page(HERE / "views" / "report.py", title="Session report"),
    st.Page(HERE / "views" / "history.py", title="History"),
    st.Page(HERE / "views" / "profile.py", title="Profile"),
])
nav.run()
