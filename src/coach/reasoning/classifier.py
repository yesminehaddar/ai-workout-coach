"""Stage 4 : reconnaissance de l'exercice avec un LSTM (PyTorch).

Entrée (30, 16) -> LSTM 2 couches x 64 -> Dense + softmax -> squat | pushup | curl | idle.
Exécuté toutes les 10 frames ; le label ne change qu'après 5 votes > 0.7.
"""


class ExerciseLSTM:  # hérite de torch.nn.Module lors de l'implémentation
    """Définition du réseau (utilisée par training/train_classifier.py)."""

    def __init__(self, input_features: int = 16, hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 4, dropout: float = 0.3):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ExerciseClassifier:
    """Charge le modèle TorchScript exporté et lisse les prédictions."""

    def __init__(self, model_path: str, min_confidence: float = 0.7, votes_to_switch: int = 5):
        raise NotImplementedError

    def predict(self, window) -> tuple[str, float]:
        raise NotImplementedError
