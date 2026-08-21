import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def remove_outliers(
    X_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.95
) -> tuple[pd.DataFrame, pd.Series]:
    threshold = y_train.quantile(threshold)
    mask = y_train < threshold
    X_train = X_train[mask]
    y_train = y_train[mask]

    return X_train, y_train


def features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return df


def features_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[["PU_DO", "trip_distance"]]
    y = df["duration"]
    return X, y


def feature_vectorization(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, DictVectorizer]:
    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train.to_dict(orient="records"))
    X_test = dv.transform(X_test.to_dict(orient="records"))
    return X_train, X_test, dv
