"""Éléments partagés par les pages Streamlit : configs, base de données, style, clips de démo."""
from pathlib import Path

import pandas as pd
import streamlit as st

from coach.config import load_config
from coach.db.database import init_db
from coach.db.repository import SessionRepository

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data" / "raw" / "fitness_form"

# Clips de profil où le compteur tombe juste (scripts/eval_rep_counter.py) et dont le contraste
# bonne / mauvaise forme est net (aucune règle déclenchée sur les « good »). Pompes : même personne.
DEMO_CLIPS = {
    "Squat · good form": "subject_010_squat_good_side.mp4",
    "Squat · bad form": "subject_009_squat_bad_side.mp4",
    "Push-up · good form": "subject_006_push_up_good_side.mp4",
    "Push-up · bad form": "subject_006_push_up_bad_side.mp4",
    "Bicep curl · good form": "subject_017_bicep_curl_good_side.mp4",
    "Bicep curl · bad form": "subject_020_bicep_curl_bad_side.mp4",
}

EXERCISE_LABELS = {"squat": "Squat", "pushup": "Push-up", "curl": "Bicep curl"}


def exercise_names(codes: str) -> str:
    """« squat, pushup » -> « Squat, Push-up » (liste stockée en base)."""
    return ", ".join(EXERCISE_LABELS.get(c.strip(), c.strip()) for c in codes.split(",") if c.strip())
FAULT_LABELS = {
    "SQUAT_LEAN": "Leaning forward", "SQUAT_SHALLOW": "Not deep enough",
    "SQUAT_KNEES_FORWARD": "Knees past toes", "SQUAT_FAST": "Too fast",
    "PUSHUP_SAG": "Hips sagging", "PUSHUP_SHALLOW": "Not low enough",
    "PUSHUP_HANDS_FORWARD": "Hands too far forward",
    "CURL_BODY_SWING": "Swinging the body", "CURL_SWING": "Elbows moving",
    "CURL_PARTIAL": "Partial range", "CURL_FAST": "Too fast",
}

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.card {background: var(--secondary-background-color); border-radius: 14px; padding: 14px 18px;
       margin-bottom: 12px; border: 1px solid rgba(128,128,128,.15);}
.card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65;}
.card .value {font-size: 2.1rem; font-weight: 700; line-height: 1.15;}
.card .sub {font-size: .85rem; opacity: .7;}
.tip {border-radius: 14px; padding: 16px 18px; font-size: 1.25rem; font-weight: 600;
      margin-bottom: 12px; text-align: center;}
.tip.warn {background: rgba(255,170,0,.16); border: 1px solid rgba(255,170,0,.5);}
.tip.good {background: rgba(46,204,113,.16); border: 1px solid rgba(46,204,113,.5);}
.tip.idle {background: var(--secondary-background-color); opacity: .6; font-weight: 500;}
.hero {font-size: 1.05rem; opacity: .8; margin-top: -0.6rem; margin-bottom: 1.2rem;}
[data-testid="stImage"], [data-testid="stImageContainer"] {margin-left: auto; margin-right: auto;}
[data-testid="stImage"] img {border-radius: 14px;}
[data-testid="stCaptionContainer"] {text-align: center;}
</style>
"""


def setup_page() -> None:
    """Style commun ; le titre et l'icône de l'onglet viennent de st.navigation (streamlit_app.py)."""
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def configs() -> tuple[dict, dict]:
    return load_config("app"), load_config("rules")


@st.cache_resource
def repository() -> SessionRepository:
    return SessionRepository(init_db())


def card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def score_color(score: float | None) -> str:
    if score is None:
        return "inherit"
    return "#2ecc71" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"


def render_report(report: dict) -> None:
    """Bilan d'une séance (dict de SessionRepository.session_report)."""
    sets = [s for s in report["sets"] if s["reps"] or s["partial"]]
    total = sum(s["reps"] for s in sets)
    avg = report.get("avg_score")
    cols = st.columns(4)
    cols[0].markdown(card("Total reps", str(total)), unsafe_allow_html=True)
    cols[1].markdown(card("Partial reps", str(sum(s["partial"] for s in sets))), unsafe_allow_html=True)
    cols[2].markdown(card("Average form score",
                          f'<span style="color:{score_color(avg)}">{avg:.0f}</span>' if avg is not None else "–",
                          "out of 100"), unsafe_allow_html=True)
    dur = report.get("duration_s")
    cols[3].markdown(card("Duration", f"{dur:.0f} s" if dur else "–"), unsafe_allow_html=True)
    for s in sets:
        st.subheader(EXERCISE_LABELS.get(s["exercise"], s["exercise"]))
        left, right = st.columns([3, 2])
        with left:
            if s["scores"]:
                st.caption("Form score per rep")
                scores = pd.DataFrame({"score": s["scores"]}, index=pd.RangeIndex(1, len(s["scores"]) + 1, name="rep"))
                st.bar_chart(scores, height=220, color="#4c8bf5", x_label="rep", y_label="score")
        with right:
            st.caption("Most frequent issues")
            avg_set = sum(s["scores"]) / len(s["scores"]) if s["scores"] else None
            if s["faults"]:
                for code, n in sorted(s["faults"].items(), key=lambda kv: -kv[1]):
                    st.markdown(f"- **{FAULT_LABELS.get(code, code)}** — {n} rep{'s' if n > 1 else ''}")
            elif avg_set is not None and avg_set < 50:
                st.warning("No specific mistake matched the rules, but the overall form score is low: "
                           "focus on slow, controlled, full-range reps.")
            else:
                st.success("No form issues detected. Great job!")
