"""Page principale : vue webcam en direct (streamlit-webrtc), squelette, compteur, conseil.

Lancer : streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="AI Workout Coach", layout="wide")
st.title("AI Workout Coach")
st.info("Architecture en place : la vue live sera branchée sur coach.pipeline.CoachPipeline.")
# TODO: webrtc_streamer(...) + CoachPipeline.process(frame)
