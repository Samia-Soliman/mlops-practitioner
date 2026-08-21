import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import Settings


def load_data(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.data_path)


def split_data(
    df: pd.DataFrame, y: pd.Series, settings: Settings
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, X_val, y_train, y_val = train_test_split(
        df,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    return X_train, y_train, X_val, y_val
