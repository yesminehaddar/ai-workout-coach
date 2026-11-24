"""Éléments partagés par les vues Streamlit : configs, base de données, style, bilan de séance."""
from pathlib import Path

import pandas as pd
import streamlit as st

from coach.config import load_config
from coach.db.database import init_db
from coach.db.repository import ProfileRepository, SessionRepository
from coach.feedback.voice import Voice

ROOT = Path(__file__).resolve().parents[1]

EXERCISE_LABELS = {"squat": "Squat", "pushup": "Push-up", "curl": "Bicep curl"}

# Bonnes pratiques générales, affichées dans le bilan de séance sous les corrections.
BEST_PRACTICE = {
    "squat": [
        "Feet shoulder-width apart, toes turned out slightly (about 15-30°).",
        "Brace your core before each rep, chest up, back neutral, eyes forward.",
        "Hips go back and down together with the knees; knees follow the line of the toes.",
        "Go down to thighs parallel to the floor if your mobility allows, heels flat.",
        "2-3 s down, short pause, then drive up through the whole foot. Breathe in down, out up.",
    ],
    "pushup": [
        "Hands slightly wider than the shoulders, under the shoulders, fingers forward.",
        "Body in one straight line from head to heels: glutes and abs squeezed.",
        "Elbows about 45° from the body, not flared out to the sides.",
        "Lower until the chest is close to the floor, then push all the way up.",
        "Too hard? Hands on a bench or knees on the floor, keeping the same straight line.",
    ],
    "curl": [
        "Stand tall, feet hip-width, knees soft, core braced.",
        "Elbows pinned to your sides: only the forearms move.",
        "Full range: arms almost straight at the bottom, squeeze at the top.",
        "About 1-2 s up and 2-3 s down; no swinging or leaning back.",
        "Pick a weight you can lift for every rep with this form.",
    ],
}


def exercise_names(codes: str) -> str:
    """« squat, pushup » -> « Squat, Push-up » (liste stockée en base)."""
    return ", ".join(EXERCISE_LABELS.get(c.strip(), c.strip()) for c in codes.split(",") if c.strip())
FAULT_LABELS = {
    "SQUAT_LEAN": "Leaning forward", "SQUAT_SHALLOW": "Not deep enough",
    "SQUAT_KNEES_FORWARD": "Knees past toes", "SQUAT_FAST": "Too fast",
    "SQUAT_KNEES_IN": "Knees caving in", "SQUAT_NARROW": "Feet too close",
    "PUSHUP_SAG": "Hips sagging", "PUSHUP_SHALLOW": "Not low enough",
    "PUSHUP_HANDS_FORWARD": "Hands too far forward",
    "CURL_BODY_SWING": "Swinging the body", "CURL_SWING": "Elbows moving",
    "CURL_PARTIAL": "Partial range", "CURL_FAST": "Too fast",
}

INK, PAPER, WHITE, VOLT, EMBER, MOSS, LINE, MIST = (
    "#0F1411", "#F5F4EF", "#FFFFFF", "#C8F31D", "#FF6A3D", "#3C4A40", "#DEDCD3", "#C9D0CA")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp *:not([data-testid="stIconMaterial"]) {{font-family: 'Inter', sans-serif;}}
.block-container {{padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}}
h1 {{font-weight: 800; letter-spacing: -0.02em; color: {INK};}}
h2, h3 {{font-weight: 700; letter-spacing: -0.01em; color: {INK};}}
[data-testid="stSidebar"] {{border-right: 1px solid {LINE};}}

/* Boutons : principal Volt sur Ink, secondaire Ink au trait */
.stButton button[kind="primary"] {{background: {VOLT}; color: {INK}; border: 1px solid {VOLT}; font-weight: 700;}}
.stButton button[kind="primary"]:hover {{background: {INK}; color: {VOLT}; border-color: {INK};}}
.stButton button[kind="secondary"] {{background: {WHITE}; color: {INK}; border: 1px solid {INK}; font-weight: 600;}}
.stButton button[kind="secondary"]:hover {{background: {INK}; color: {WHITE};}}

/* Cartes claires */
.card {{background: {WHITE}; border: 1px solid {LINE}; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;}}
.card .label {{font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: {MOSS};}}
.card .value {{font-size: 2.2rem; font-weight: 800; line-height: 1.15; color: {INK};}}
.card .sub {{font-size: .85rem; color: {MOSS};}}
.pill {{display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .85rem;}}
.pill.good {{background: {VOLT}; color: {INK};}}
.pill.bad {{background: {EMBER}; color: {INK};}}
.fix {{background: {WHITE}; border: 1px solid {LINE}; border-left: 4px solid {EMBER}; border-radius: 10px;
       padding: 12px 16px; margin-bottom: 10px; font-size: .95rem; color: {INK};}}
.muted {{color: {MOSS};}}

/* Panneau du coach (section sombre), lisible à 2 m */
.coach {{background: {INK}; color: {WHITE}; border-radius: 16px; padding: 22px 26px; min-height: 520px;}}
.coach .eyebrow {{font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .12em; color: {MIST};}}
.coach .exercise {{font-size: 1.6rem; font-weight: 700; margin: 2px 0 10px;}}
.coach .reps {{font-size: 7.5rem; font-weight: 800; line-height: .95; color: {VOLT}; letter-spacing: -0.04em;}}
.coach .reps-sub {{color: {MIST}; font-size: .95rem; margin-bottom: 18px;}}
.coach .verdict {{display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 800; font-size: 1.15rem;
                  margin-bottom: 14px;}}
.coach .verdict.good {{background: {VOLT}; color: {INK};}}
.coach .verdict.bad {{background: {EMBER}; color: {INK};}}
.coach .cue {{font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin-bottom: 12px;}}
.coach .cue.good {{color: {VOLT};}}
.coach .cue.idle {{font-size: 1.3rem; font-weight: 500; color: {MIST};}}
.coach .how {{color: {MIST}; font-size: 1.05rem; line-height: 1.45; border-top: 1px solid {MOSS}; padding-top: 12px;}}
.coach .how b {{color: {WHITE};}}
.coach .score {{color: {MIST}; font-size: .95rem; margin-top: 16px;}}
.coach .bar {{height: 6px; background: {MOSS}; border-radius: 3px; margin-top: 6px; overflow: hidden;}}
.coach .bar span {{display: block; height: 100%; background: {VOLT};}}

[data-testid="stImage"], [data-testid="stImageContainer"] {{margin-left: auto; margin-right: auto;}}
[data-testid="stImage"] img {{border-radius: 16px;}}
[data-testid="stImageCaption"], [data-testid="stImage"] [data-testid="stCaptionContainer"] {{text-align: center;}}
[data-testid="stCaptionContainer"] {{color: {MOSS};}}
</style>
"""


def setup_page() -> None:
    """Style commun ; le titre et l'icône de l'onglet viennent de st.navigation (streamlit_app.py)."""
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def configs() -> tuple[dict, dict]:
    return load_config("app"), load_config("rules")


def fault_fixes() -> dict[str, str]:
    """Code du défaut -> comment le corriger (champ fix de backend/coach/configs/rules.yaml)."""
    return {r["code"]: r.get("fix", "") for rules in configs()[1]["form_rules"].values() for r in rules}


def classifier_available() -> bool:
    """Le mode auto a besoin du LSTM entraîné (ml/training/train_classifier.py)."""
    return (ROOT / configs()[0]["classifier"]["model_path"]).exists()


@st.cache_resource
def repository() -> SessionRepository:
    return SessionRepository(init_db())


@st.cache_resource
def profile_repository() -> ProfileRepository:
    return ProfileRepository(repository().factory)


@st.cache_resource
def voice() -> Voice:
    """Une seule voix pour toute l'appli (thread de synthèse vocale)."""
    return Voice()


def coach_panel(exercise: str, reps: str, reps_sub: str = "", verdict: str | None = None,
                cue: str | None = None, cue_kind: str = "warn", how: str | None = None,
                score: float | None = None) -> str:
    """Panneau sombre du coach : exercice, reps en très grand, verdict, consigne, correction."""
    parts = [f'<div class="coach"><div class="eyebrow">Exercise</div><div class="exercise">{exercise}</div>',
             f'<div class="eyebrow">Reps</div><div class="reps">{reps}</div>',
             f'<div class="reps-sub">{reps_sub}</div>']
    if verdict:
        text = "Good form" if verdict == "good" else "Fix your form"
        parts.append(f'<div class="verdict {verdict}">{text}</div>')
    parts.append(f'<div class="cue {cue_kind}">{cue}</div>' if cue
                 else '<div class="cue idle">Your coaching cue appears here after each rep</div>')
    if how:
        parts.append(f'<div class="how"><b>How to correct it</b><br>{how}</div>')
    if score is not None:
        parts.append(f'<div class="score">Form score {score:.0f}/100'
                     f'<div class="bar"><span style="width:{max(0, min(100, score)):.0f}%"></span></div></div>')
    return "".join(parts) + "</div>"


def card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def score_color(score: float | None) -> str:
    if score is None:
        return "inherit"
    return INK if score >= 75 else MOSS if score >= 50 else EMBER


def render_report(report: dict) -> None:
    """Bilan d'une séance (dict de SessionRepository.session_report)."""
    sets = [s for s in report["sets"] if s["reps"] or s["partial"]]
    total = sum(s["reps"] for s in sets)
    avg = report.get("avg_score")
    cols = st.columns(4)
    cols[0].markdown(card("Total reps", str(total)), unsafe_allow_html=True)
    cols[1].markdown(card("Partial reps", str(sum(s["partial"] for s in sets))), unsafe_allow_html=True)
    cols[2].markdown(card("Average form score",
                          f'<span style="color:{score_color(avg)}">{avg:.0f}</span>' if avg is not None else "-",
                          "out of 100"), unsafe_allow_html=True)
    dur = report.get("duration_s")
    cols[3].markdown(card("Duration", f"{dur:.0f} s" if dur else "-"), unsafe_allow_html=True)
    fixes = fault_fixes()
    for s in sets:
        st.subheader(EXERCISE_LABELS.get(s["exercise"], s["exercise"]))
        verdicts = s.get("verdicts", [])
        if verdicts:
            n_good = verdicts.count("good")
            st.markdown(f'<span class="pill good">{n_good} good</span>&nbsp;&nbsp;'
                        f'<span class="pill bad">{len(verdicts) - n_good} to fix</span>&nbsp;&nbsp;'
                        f'<span class="muted">out of {len(verdicts)} reps</span>', unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            if s["scores"]:
                st.caption("Form score per rep: dark = good form, orange = to fix")
                vs = verdicts or ["good"] * len(s["scores"])
                scores = pd.DataFrame(
                    {"good": [sc if v == "good" else None for sc, v in zip(s["scores"], vs)],
                     "to fix": [sc if v == "bad" else None for sc, v in zip(s["scores"], vs)]},
                    index=pd.RangeIndex(1, len(s["scores"]) + 1, name="rep"))
                st.bar_chart(scores, height=220, color=[INK, EMBER], x_label="rep", y_label="score")
        with right:
            st.caption("Most frequent issues")
            avg_set = sum(s["scores"]) / len(s["scores"]) if s["scores"] else None
            if s["faults"]:
                for code, n in sorted(s["faults"].items(), key=lambda kv: -kv[1]):
                    st.markdown(f"- **{FAULT_LABELS.get(code, code)}**: {n} rep{'s' if n > 1 else ''}")
            elif avg_set is not None and avg_set < 50:
                st.warning("No specific mistake matched the rules, but the overall form score is low: "
                           "focus on slow, controlled, full-range reps.")
            else:
                st.markdown('<span class="pill good">No form issues</span>', unsafe_allow_html=True)
        if s["faults"]:
            st.markdown("**How to correct your form**")
            for code, _ in sorted(s["faults"].items(), key=lambda kv: -kv[1]):
                if fixes.get(code):
                    st.markdown(f'<div class="fix"><b>{FAULT_LABELS.get(code, code)}</b><br>{fixes[code]}</div>',
                                unsafe_allow_html=True)
        best = BEST_PRACTICE.get(s["exercise"])
        if best:
            with st.expander(f"Best practice for the {EXERCISE_LABELS.get(s['exercise'], s['exercise']).lower()}"):
                st.markdown("\n".join(f"- {line}" for line in best))
