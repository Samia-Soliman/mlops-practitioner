import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"{func.__name__} took {duration:.6f} seconds")
        return result

    return wrapper


class DurationPredictor:

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model: Any = None
        self.load()

    def load(self) -> None:
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        return float(self.model.predict([features])[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        return [float(pred) for pred in self.model.predict(features_list)]
