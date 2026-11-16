"""Historique des séances par date, pour suivre la progression."""
import pandas as pd
import streamlit as st

from coach.ui import card, exercise_names, repository, setup_page

setup_page()
st.title("📈 History")

sessions = repository().list_sessions()
if not sessions:
    st.info("No session yet. Run one on the **Live coach** page.")
    st.stop()

df = pd.DataFrame(sessions)
cols = st.columns(3)
cols[0].markdown(card("Sessions", str(len(df))), unsafe_allow_html=True)
cols[1].markdown(card("Total reps", str(int(df.reps.sum()))), unsafe_allow_html=True)
best = df.avg_score.max()
cols[2].markdown(card("Best average score", f"{best:.0f}" if pd.notna(best) else "–", "out of 100"),
                 unsafe_allow_html=True)

scored = df.dropna(subset=["avg_score"])
if len(scored) > 1:
    st.subheader("Average form score over time")
    trend = scored.set_index(pd.Index(scored["id"], name="session"))["avg_score"]
    st.line_chart(trend, height=260, color="#4c8bf5", x_label="session", y_label="average score")

st.subheader("All sessions")
table = df.assign(exercises=df.exercises.map(exercise_names), date=df.started_at.dt.strftime("%d %b %Y, %H:%M"),
                  duration=df.duration_s.round(0), score=df.avg_score.round(0))
st.dataframe(table[["id", "date", "exercises", "reps", "score", "duration"]].iloc[::-1],
             hide_index=True, width="stretch",
             column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100,
                                                                     format="%d"),
                            "duration": st.column_config.NumberColumn("duration (s)")})
