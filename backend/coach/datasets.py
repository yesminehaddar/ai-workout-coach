"""Jeux de données d'entraînement : Mendeley (all.parquet) et QEVD-FIT-300k (qevd.parquet).

Les deux fichiers ont les mêmes colonnes de features (ml/data/extract_keypoints.py et
ml/data/extract_qevd.py). Ce module gère ce qui diffère :

- Personne : Mendeley a un sujet -> groupe « m<sujet> ». QEVD n'a pas d'identifiant de personne
  -> groupe « q<vidéo> » ; une même personne peut apparaître dans plusieurs clips, les scores
  mesurés sur QEVD sont donc un peu optimistes.
- Séparation train / val / test : Mendeley par sujet (5 test, 3 validation, tirage fixe) ;
  QEVD garde sa séparation train / test, et 10 % des clips train servent de validation.
- Forme : Mendeley dit bonne / mauvaise par clip ; QEVD nomme les défauts. qevd_label() en tire
  quality ("good" / "bad") et la liste des défauts, ou None pour un clip inexploitable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SOURCES = {"mendeley": "all.parquet", "qevd": "qevd.parquet"}

# Étiquette QEVD -> défaut. Les autres étiquettes décrivent la variante (« wide », « 90 degrees »,
# « plie »...) et ne sont pas des défauts.
QEVD_FAULTS = {
    "squat": {"shallow": "shallow", "insufficient": "shallow", "back not straight": "back_not_straight",
              "knees over toes": "knees_over_toes", "narrow": "narrow_stance"},
    "push_up": {"shallow": "shallow", "insufficient": "shallow", "elevated hips": "hips_high",
                "butt sinking": "hips_sag", "child pose cheat": "child_pose_cheat", "sloppy": "sloppy",
                "fast": "fast", "head down": "head_down", "moving head": "head_moving",
                "side to side": "side_to_side"},
}
# Clips où l'exercice n'est pas fait normalement (attente, arrêt, personne hors champ).
QEVD_UNUSABLE = {
    "squat": {"hold", "starting late", "not visible"},
    "push_up": {"waiting position", "early stopping", "not visible", "head moving in an out of frame",
                "legs on the floor"},
}
QEVD_PREFIX = {"squat": "squats - ", "push_up": "pushups - "}
TEST_SUBJECTS, VAL_SUBJECTS, QEVD_VAL_SHARE = 5, 3, 0.10


def qevd_label(labels: str, exercise: str) -> tuple[str | None, list[str]]:
    """« squats - shallow | squats - wide » -> ("bad", ["shallow"]). (None, []) si inexploitable."""
    prefix = QEVD_PREFIX[exercise]
    tags = {x[len(prefix):] for x in str(labels).split(" | ") if x.startswith(prefix)}
    if tags & QEVD_UNUSABLE[exercise]:
        return None, []
    faults = sorted({QEVD_FAULTS[exercise][t] for t in tags if t in QEVD_FAULTS[exercise]})
    return ("bad" if faults else "good"), faults


def available(processed: Path) -> list[str]:
    """Sources dont le fichier Parquet existe dans processed."""
    return [s for s, f in SOURCES.items() if (processed / f).exists()]


def parse_sources(arg: str | None, processed: Path) -> list[str]:
    """--sources "mendeley,qevd" (ou rien = toutes celles disponibles), avec un message clair."""
    have = available(processed)
    wanted = [s.strip() for s in arg.split(",")] if arg else have
    unknown = [s for s in wanted if s not in SOURCES]
    if unknown:
        raise SystemExit(f"Source inconnue : {unknown} (possibles : {list(SOURCES)})")
    missing = [s for s in wanted if s not in have]
    if missing or not wanted:
        raise SystemExit(f"Fichiers manquants dans {processed} : "
                         f"{[SOURCES[s] for s in missing] or list(SOURCES.values())}. "
                         "Mendeley : ml/data/extract_keypoints.py ; QEVD : ml/data/extract_qevd.py")
    if "mendeley" not in wanted:
        print("ATTENTION : sans Mendeley, pas de curl ni de vue de profil dans l'entraînement.")
    return wanted


def add_stance_width(df: pd.DataFrame) -> pd.DataFrame:
    """stance_width (écart des chevilles / largeur d'épaules) depuis les 33 points stockés, calculé
    comme coach.features.angles.compute_features (x multiplié par largeur / hauteur de l'image)."""
    a = df["width"] / df["height"]
    ankle = np.hypot((df["lm27_x"] - df["lm28_x"]) * a, df["lm27_y"] - df["lm28_y"])
    shoulder = np.hypot((df["lm11_x"] - df["lm12_x"]) * a, df["lm11_y"] - df["lm12_y"])
    return df.assign(stance_width=(ankle / shoulder).where(shoulder > 1e-6, 1.0))  # 1.0 = neutre


STANCE_COLUMNS = ["width", "height", "lm11_x", "lm11_y", "lm12_x", "lm12_y", "lm27_x", "lm27_y", "lm28_x", "lm28_y"]


def load_frames(processed: Path, sources: list[str], columns: list[str]) -> pd.DataFrame:
    """Frames des sources demandées, avec en plus : source, group, split, quality, faults.

    split vaut "train", "val" ou "test" (même tirage que l'entraînement d'origine pour Mendeley).
    """
    parts = []
    for src in sources:
        path = processed / SOURCES[src]
        schema = set(pq.read_schema(path).names)
        extra = ["subject", "quality"] if src == "mendeley" else ["labels", "split"]
        if "stance_width" in columns and "stance_width" not in schema:
            extra += STANCE_COLUMNS
        df = pd.read_parquet(path, columns=[c for c in dict.fromkeys(columns + extra) if c in schema])
        if "stance_width" in columns and "stance_width" not in df and set(STANCE_COLUMNS) <= set(df):
            df = add_stance_width(df).drop(columns=[c for c in STANCE_COLUMNS if c not in columns])
        df["source"] = src
        if src == "mendeley":
            df["group"] = "m" + df["subject"].astype(int).astype(str)
            df["split"] = _subject_split(df["subject"])
            df["faults"] = ""
        else:
            df["group"] = "q" + df["video"].astype(str)
            df["split"] = _qevd_split(df)
            lab = {v: qevd_label(g["labels"].iloc[0], g["exercise"].iloc[0])
                   for v, g in df.groupby("video")}
            df["quality"] = df["video"].map({v: q for v, (q, _) in lab.items()})
            df["faults"] = df["video"].map({v: ",".join(f) for v, (_, f) in lab.items()})
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _subject_split(subject: pd.Series) -> pd.Series:
    """Tirage de ml/training/train_classifier.py d'origine : 5 sujets test, 3 validation."""
    perm = np.random.default_rng(0).permutation(sorted(subject.unique()))
    test, val = set(perm[:TEST_SUBJECTS]), set(perm[TEST_SUBJECTS:TEST_SUBJECTS + VAL_SUBJECTS])
    return subject.map(lambda s: "test" if s in test else "val" if s in val else "train")


def _qevd_split(df: pd.DataFrame) -> pd.Series:
    """Séparation QEVD d'origine ; 10 % des vidéos train (tirage fixe) passent en validation."""
    per_video = df.groupby("video")["split"].first()
    train = sorted(per_video[per_video == "train"].index)
    rng = np.random.default_rng(0)
    val = set(rng.choice(train, int(len(train) * QEVD_VAL_SHARE), replace=False)) if train else set()
    mapping = {v: "test" if s == "test" else "val" if v in val else "train" for v, s in per_video.items()}
    return df["video"].map(mapping)
