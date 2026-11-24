"""Profil sportif : âge, taille, poids, poids cible, genre, niveau, matériel.

Lu par le chatbot « Plan my session » pour adapter la séance proposée.
"""
import streamlit as st

from coach.planner import EQUIPMENT, EXPERIENCE, GENDERS, AthleteProfile
from coach.planner.profile import LIMITS
from ui import card, profile_repository, setup_page

setup_page()
st.title("Profile")
st.markdown('<p class="muted">Used by the Gym Bro chatbot to plan your sessions. '
            'Stored only in the local database on this computer.</p>', unsafe_allow_html=True)

saved = profile_repository().get()
default = saved or {"age": 25, "gender": GENDERS[0], "height_cm": 175.0, "weight_kg": 75.0,
                    "target_weight_kg": 75.0, "experience": EXPERIENCE[0], "equipment": EQUIPMENT[0]}


def _index(options: tuple, value: str) -> int:
    return options.index(value) if value in options else 0


with st.form("profile"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", *LIMITS["age"], int(default["age"]), step=1)
    gender = c2.selectbox("Gender", GENDERS, index=_index(GENDERS, default["gender"]))
    height = c1.number_input("Height (cm)", *LIMITS["height_cm"], float(default["height_cm"]), step=1.0,
                             format="%.0f")
    weight = c2.number_input("Weight (kg)", *LIMITS["weight_kg"], float(default["weight_kg"]), step=0.5,
                             format="%.1f")
    target = c1.number_input("Target weight (kg)", *LIMITS["target_weight_kg"],
                             float(default["target_weight_kg"]), step=0.5, format="%.1f")
    experience = c2.selectbox("Training experience", EXPERIENCE,
                              index=_index(EXPERIENCE, default["experience"]))
    equipment = c1.selectbox("Equipment", EQUIPMENT, index=_index(EQUIPMENT, default["equipment"]))
    submitted = st.form_submit_button("Save profile", type="primary")

if submitted:
    try:
        profile = AthleteProfile(age=int(age), gender=gender, height_cm=float(height), weight_kg=float(weight),
                                 target_weight_kg=float(target), experience=experience, equipment=equipment)
    except ValueError as e:
        st.error(str(e).capitalize())
    else:
        profile_repository().save(profile.to_dict())
        saved = profile.to_dict()
        st.success("Profile saved.")

if saved:
    p = AthleteProfile.from_dict(saved)
    cols = st.columns(3)
    cols[0].markdown(card("BMI", f"{p.bmi:.1f}", "weight / height²"), unsafe_allow_html=True)
    cols[1].markdown(card("Goal", p.goal.capitalize(), f"{p.weight_delta_kg:+.1f} kg to target"),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Level", p.experience.capitalize(), p.equipment), unsafe_allow_html=True)
    st.page_link("views/planner.py", label="Plan today's session with Gym Bro", icon=":material/chat:")
