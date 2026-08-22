import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from logging import getLogger

logger = getLogger(__name__)


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"{func.__name__} took {duration:.6f} seconds")
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
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def validate_features(self, features: dict[str, Any]) -> None:

        if "trip_distance" not in features:
            logger.error("Validation rejection | missing trip_distance")
            raise ValueError("trip_distance is required.")

        if features["trip_distance"] > 100:
            logger.warning(
                "Input outside training range | trip_distance=%s",
                features["trip_distance"],
            )

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        if self.model is None:
            logger.error("Model has not been loaded.")
            raise RuntimeError("Model has not been loaded.")
        logger.debug(f"{self.__class__.__name__} Predicting for features: {features}")
        self.validate_features(features)

        return float(self.model.predict([features])[0])

    @timed
    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        if self.model is None:
            logger.error("Model has not been loaded.")
            raise RuntimeError("Model has not been loaded.")
        logger.debug(
            f"{self.__class__.__name__} Predicting for features list: {features_list}"
        )
        for features in features_list:
            self.validate_features(features)
        return [float(pred) for pred in self.model.predict(features_list)]
