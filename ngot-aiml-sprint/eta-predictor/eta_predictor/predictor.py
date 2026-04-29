import joblib
import numpy as np
from pathlib import Path

class Predictor:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.is_loaded = False
        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        self.is_loaded = True

    def predict(self, features: list[float]):
        # If model not loaded, return simple heuristic
        if not self.is_loaded:
            dist, weight, hour, is_truck = features
            eta = dist / 50 * 60 + weight * 0.05 + (1 - is_truck) * 5
            ci = 0.1 * eta
            return eta, eta - ci, eta + ci
        arr = np.array(features).reshape(1, -1)
        pred = self.model.predict(arr)
        return float(pred[0]), float(pred[0]) - 1.0, float(pred[0]) + 1.0
