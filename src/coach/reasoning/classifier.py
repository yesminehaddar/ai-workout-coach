"""Stage 4 : reconnaissance de l'exercice avec un LSTM (PyTorch).

Entrée (30, 16) -> LSTM 2 couches x 64 -> Dense -> squat | pushup | curl | idle.
« idle » = tout le reste (dans le dataset : développé épaules, triceps, rowing, abdos).
La normalisation (moyenne / écart-type du train) est stockée dans le modèle : on lui donne
les features brutes de compute_features + lissage, comme dans l'appli.
Exécuté toutes les 10 frames ; le label ne change qu'après 5 votes > 0.7.
"""
from collections import deque

import numpy as np
import torch
from torch import nn

CLASSES = ["squat", "pushup", "curl", "idle"]


class ExerciseLSTM(nn.Module):
    """Définition du réseau (utilisée par training/train_classifier.py)."""

    def __init__(self, input_features: int = 16, hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_features))
        self.register_buffer("std", torch.ones(input_features))
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, 30, 16) features brutes -> logits (batch, num_classes)."""
        x = torch.nan_to_num((x - self.mean) / self.std)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class ExerciseClassifier:
    """Charge le modèle TorchScript exporté et lisse les prédictions."""

    def __init__(self, model_path: str, min_confidence: float = 0.7, votes_to_switch: int = 5):
        self.model = torch.jit.load(model_path, map_location="cpu").eval()
        self.min_confidence = min_confidence
        self.votes = deque(maxlen=votes_to_switch)
        self.label = "idle"

    @torch.no_grad()
    def predict(self, window) -> tuple[str, float]:
        """window : (30, 16). Renvoie le label stable et la confiance de la dernière prédiction.
        Le label stable ne change qu'après votes_to_switch prédictions identiques > min_confidence."""
        x = torch.as_tensor(np.asarray(window, dtype=np.float32)).unsqueeze(0)
        p = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = float(p.max()), int(p.argmax())
        self.votes.append(CLASSES[idx] if conf >= self.min_confidence else None)
        if len(self.votes) == self.votes.maxlen and len(set(self.votes)) == 1 and self.votes[0]:
            self.label = self.votes[0]
        return self.label, conf
