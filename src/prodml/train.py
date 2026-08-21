import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

from prodml.config import Settings
from prodml.data import load_data, split_data
from prodml.features import (
    feature_vectorization,
    features_engineering,
    features_selection,
    remove_outliers,
)

df = load_data(Settings())
df = features_engineering(df)
X, y = features_selection(df)
X_train, y_train, X_val, y_val = split_data(X, y, Settings())
X_train, y_train = remove_outliers(X_train, y_train)
X_train, X_val, dv = feature_vectorization(X_train, X_val)

rf = RandomForestRegressor(n_estimators=100, random_state=Settings().random_state)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_rmse = root_mean_squared_error(y_val, rf_pred)
rf_mae = mean_absolute_error(y_val, rf_pred)
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")

pipeline = Pipeline(
    [
        ("vectorizer", dv),
        ("model", rf),
    ]
)

with open(Settings().model_path, "wb") as f:
    pickle.dump(pipeline, f)
