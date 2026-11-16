"""Point d'entrée de l'application : navigation entre les trois pages.

Lancer : python -m streamlit run app/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="AI Workout Coach", page_icon="🏋️", layout="wide")
nav = st.navigation([
    st.Page(HERE / "live.py", title="Live coach", icon="🏋️", default=True),
    st.Page(HERE / "report.py", title="Session report", icon="📋"),
    st.Page(HERE / "history.py", title="History", icon="📈"),
])
nav.run()
