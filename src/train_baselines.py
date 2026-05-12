import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import (
    MERGED_DATASET_PARQUET,
    FEATURE_COLS,
    TARGET_COL,
    TABLES,
    MODELS,
    TRAIN_END,
    VAL_END,
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def peak_mae(y_true, y_pred, q=0.99):
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    return mean_absolute_error(y_true[mask], y_pred[mask])

def train_val_test_split(df):
    train = df[df["timestamp"] <= TRAIN_END].copy()
    val = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)].copy()
    test = df[df["timestamp"] > VAL_END].copy()
    return train, val, test

def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Top1pct_MAE": peak_mae(y_true, y_pred, q=0.99),
    }

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MERGED_DATASET_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train, val, test = train_val_test_split(df)
    trainval = pd.concat([train, val], axis=0)

    results = []

    # baseline 1: yesterday same hour
    y_pred_24 = test["lag_24"].values
    results.append(evaluate("seasonal_naive_24", test[TARGET_COL].values, y_pred_24))

    # baseline 2: last week same hour
    y_pred_168 = test["lag_168"].values
    results.append(evaluate("seasonal_naive_168", test[TARGET_COL].values, y_pred_168))

    # RandomForest baseline
    X_train = trainval[FEATURE_COLS].values
    y_train = trainval[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    results.append(evaluate("random_forest", y_test, y_pred_rf))

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(TABLES / "baseline_metrics.csv", index=False)

    joblib.dump(scaler, MODELS / "baseline_scaler.pkl")
    joblib.dump(rf, MODELS / "random_forest.pkl")

    pred_df = test[["timestamp", TARGET_COL]].copy()
    pred_df["pred_naive_24"] = y_pred_24
    pred_df["pred_naive_168"] = y_pred_168
    pred_df["pred_random_forest"] = y_pred_rf
    pred_df.to_csv(TABLES / "baseline_predictions_test.csv", index=False)

    print(metrics_df)

if __name__ == "__main__":
    main()