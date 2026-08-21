from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Path("data/green_tripdata_2026-01.parquet")
    model_path: Path = Path("models/model.pkl")
    test_size: float = 0.2
    random_state: int = 42
