"""Jeux d'entraînement : étiquettes QEVD -> forme et défauts, séparations stables, deux sources."""
import pandas as pd
import pytest

from coach.datasets import load_frames, parse_sources, qevd_label


def test_qevd_clean_squat_is_good():
    assert qevd_label("squats - no obvious issue | squats - shoulder-width | squats - 90 degrees",
                      "squat") == ("good", [])


def test_qevd_faults_are_named_and_merged():
    q, faults = qevd_label("squats - shallow | squats - insufficient | squats - knees over toes", "squat")
    assert q == "bad" and faults == ["knees_over_toes", "shallow"]


def test_qevd_unusable_clip_is_dropped():
    assert qevd_label("squats - hold | squats - shoulder-width", "squat") == (None, [])
    assert qevd_label("pushups - waiting position", "push_up") == (None, [])


def test_qevd_pushup_faults():
    assert qevd_label("pushups - butt sinking | pushups - wide", "push_up") == ("bad", ["hips_sag"])


def _write_sources(tmp_path, n_subjects=10):
    rows = [{"video": f"subject_{s:03d}_squat_good_side.mp4", "subject": s, "exercise": "squat",
             "quality": "good", "view": "side", "frame": 0, "t": 0.0, "knee": 170.0}
            for s in range(1, n_subjects + 1)]
    pd.DataFrame(rows).to_parquet(tmp_path / "all.parquet")
    q = [{"video": f"{i:08d}.mp4", "exercise": "squat", "view": "front", "frame": 0, "t": 0.0,
          "knee": 160.0, "split": "test" if i < 5 else "train",
          "labels": "squats - shallow" if i % 2 else "squats - no obvious issue"} for i in range(40)]
    pd.DataFrame(q).to_parquet(tmp_path / "qevd.parquet")


def test_load_frames_combines_sources_with_splits(tmp_path):
    _write_sources(tmp_path)
    df = load_frames(tmp_path, ["mendeley", "qevd"], ["video", "exercise", "view", "frame", "t", "knee"])
    m, q = df[df.source == "mendeley"], df[df.source == "qevd"]
    assert m.groupby("split").group.nunique().to_dict() == {"test": 5, "train": 2, "val": 3}
    assert set(q.loc[q.video < "00000005", "split"]) == {"test"}
    assert (q.split == "val").sum() == 3  # 10 % des 35 clips train
    assert q.loc[q.video == "00000001.mp4", "quality"].item() == "bad"
    assert q.loc[q.video == "00000001.mp4", "faults"].item() == "shallow"
    again = load_frames(tmp_path, ["qevd"], ["video", "exercise", "view", "frame", "t", "knee"])
    assert again.split.tolist() == q.split.tolist()  # tirage fixe


def test_parse_sources_reports_missing_file(tmp_path):
    _write_sources(tmp_path)
    (tmp_path / "all.parquet").unlink()
    assert parse_sources(None, tmp_path) == ["qevd"]
    with pytest.raises(SystemExit, match="all.parquet"):
        parse_sources("mendeley,qevd", tmp_path)
