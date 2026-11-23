"""Point d'entrée de l'application : navigation entre les trois pages.

Lancer : python -m streamlit run frontend/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent  # absolu : marche quel que soit le dossier de lancement
st.set_page_config(page_title="Workout Coach", page_icon=":material/fitness_center:", layout="wide")
nav = st.navigation([
    st.Page(HERE / "views" / "live.py", title="Live coach", default=True),
    st.Page(HERE / "views" / "report.py", title="Session report"),
    st.Page(HERE / "views" / "history.py", title="History"),
])
nav.run()
