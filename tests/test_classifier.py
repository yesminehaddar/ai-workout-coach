"""Tests du classifieur : forme de sortie, export TorchScript, vote avant de changer de label."""
import numpy as np
import torch

from coach.reasoning.classifier import CLASSES, ExerciseClassifier, ExerciseLSTM


def test_lstm_output_shape_and_nan():
    m = ExerciseLSTM().eval()
    x = torch.randn(4, 30, 16)
    x[0, 3, 2] = float("nan")
    out = m(x)
    assert out.shape == (4, len(CLASSES)) and torch.isfinite(out).all()


def test_label_switches_only_after_votes(tmp_path):
    m = ExerciseLSTM().eval()
    with torch.no_grad():  # force la classe "curl" avec une forte confiance
        m.head[1].weight.zero_()
        m.head[1].bias.copy_(torch.tensor([0.0, 0.0, 10.0, 0.0]))
    path = tmp_path / "m.pt"
    torch.jit.script(m).save(str(path))
    clf = ExerciseClassifier(str(path), votes_to_switch=5)
    w = np.zeros((30, 16), dtype=np.float32)
    labels = [clf.predict(w)[0] for _ in range(5)]
    assert labels[:4] == ["idle"] * 4 and labels[4] == "curl"
