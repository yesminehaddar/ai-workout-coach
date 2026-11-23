"""Clips de démonstration proposés par l'interface et la vidéo de démo (seulement ceux présents)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data" / "raw" / "fitness_form"
QEVD_CLIPS = ROOT / "data" / "raw" / "qevd_fit300k" / "clips"  # ml/data/qevd_split_zip.py build

# Clips de profil où le compteur tombe juste (ml/evaluation/eval_rep_counter.py) et dont le contraste
# bonne / mauvaise forme est net (aucune règle déclenchée sur les « good »). Pompes : même personne.
_MENDELEY_DEMO = {
    "Squat · good form": CLIPS / "subject_010_squat_good_side.mp4",
    "Squat · bad form": CLIPS / "subject_009_squat_bad_side.mp4",
    "Push-up · good form": CLIPS / "subject_006_push_up_good_side.mp4",
    "Push-up · bad form": CLIPS / "subject_006_push_up_bad_side.mp4",
    "Bicep curl · good form": CLIPS / "subject_017_bicep_curl_good_side.mp4",
    "Bicep curl · bad form": CLIPS / "subject_020_bicep_curl_bad_side.mp4",
}
# Clips QEVD-FIT-300k (webcam de face, à la maison), choisis parmi les 758 lisibles : étiquette
# QEVD nette et le coach rend le même verdict (bonne forme : 2 reps propres ; peu profond : toutes
# les reps signalées). Attention : les seuils sont réglés de profil ; de face, l'angle du genou
# paraît plus ouvert (médiane en bas : 89° sur les clips propres, 123° sur les clips « shallow »).
_QEVD_DEMO = {
    "QEVD squat (front) · good form": QEVD_CLIPS / "00074582.mp4",
    "QEVD squat (front) · bad form: shallow and fast": QEVD_CLIPS / "00143794.mp4",
    "QEVD squat (front) · bad form: shallow": QEVD_CLIPS / "00144112.mp4",
}
# Seuls les clips présents sur ce PC sont proposés.
DEMO_CLIPS = {k: v for k, v in {**_MENDELEY_DEMO, **_QEVD_DEMO}.items() if v.exists()}
