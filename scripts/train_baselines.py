import argparse
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "urban_grid_climate_storage_matplotlib"),
)
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridclimate.config import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train config-driven historical baseline load forecasting models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def configured_path(config: dict[str, Any], key: str) -> Path:
    try:
        return resolve_project_path(config["paths"][key])
    except KeyError as exc:
        raise KeyError(f"Missing paths.{key} in config.") from exc


def parse_time(value: str) -> pd.Timestamp:
    return pd.to_datetime(value)


def split_dataset(df: pd.DataFrame, config: dict[str, Any], datetime_col: str) -> dict[str, pd.DataFrame]:
    split = config["split"]
    ranges = {
        "train": (split["train_start"], split["train_end"]),
        "validation": (split["validation_start"], split["validation_end"]),
        "test": (split["test_start"], split["test_end"]),
    }

    out: dict[str, pd.DataFrame] = {}
    for name, (start, end) in ranges.items():
        mask = (df[datetime_col] >= parse_time(start)) & (df[datetime_col] <= parse_time(end))
        out[name] = df.loc[mask].copy()
    return out


def check_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {label}: {missing}")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(
    split_name: str,
    model_name: str,
    segment_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    valid = y_true.notna() & y_pred.notna()
    y_true_arr = y_true.loc[valid].to_numpy(dtype=float)
    y_pred_arr = y_pred.loc[valid].to_numpy(dtype=float)

    if len(y_true_arr) == 0:
        return {
            "split": split_name,
            "model": model_name,
            "segment": segment_name,
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
        }

    return {
        "split": split_name,
        "model": model_name,
        "segment": segment_name,
        "n": int(len(y_true_arr)),
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": rmse(y_true_arr, y_pred_arr),
        "MAPE": mape(y_true_arr, y_pred_arr),
        "R2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else np.nan,
    }


def segment_masks(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    temp_col = config["columns"]["temperature"]
    load_col = config["columns"]["load"]
    peak_threshold = df[load_col].quantile(0.95)

    return {
        "overall": pd.Series(True, index=df.index),
        "summer_JJA": df["month"].isin([6, 7, 8]),
        "winter_DJF": df["month"].isin([12, 1, 2]),
        "hot_hours_temp_ge_30": df[temp_col] >= 30,
        "hot_hours_temp_ge_33": df[temp_col] >= 33,
        "peak_top_5_percent_load": df[load_col] >= peak_threshold,
        "weekday": df["is_weekend"] == 0,
        "weekend": df["is_weekend"] == 1,
    }


def evaluate_predictions(
    split_name: str,
    df: pd.DataFrame,
    prediction_columns: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    load_col = config["columns"]["load"]
    rows = []
    masks = segment_masks(df, config)

    for model_name, pred_col in prediction_columns.items():
        for segment_name, mask in masks.items():
            rows.append(
                metric_row(
                    split_name=split_name,
                    model_name=model_name,
                    segment_name=segment_name,
                    y_true=df.loc[mask, load_col],
                    y_pred=df.loc[mask, pred_col],
                )
            )
    return rows


def build_models(config: dict[str, Any]) -> dict[str, tuple[Any, list[str]]]:
    baseline_config = config["models"]["baseline"]
    feature_sets = baseline_config["feature_sets"]
    ridge_alpha = float(baseline_config.get("ridge", {}).get("alpha", 1.0))
    hgb_config = baseline_config.get("hist_gradient_boosting", {})

    ridge_climate = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha)),
        ]
    )
    ridge_forecast = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha)),
        ]
    )

    hgb_kwargs = {
        "max_iter": int(hgb_config.get("max_iter", 300)),
        "learning_rate": float(hgb_config.get("learning_rate", 0.05)),
        "max_leaf_nodes": int(hgb_config.get("max_leaf_nodes", 31)),
        "l2_regularization": float(hgb_config.get("l2_regularization", 0.0)),
        "random_state": int(hgb_config.get("random_state", 42)),
    }

    return {
        "ridge_climate": (ridge_climate, feature_sets["climate"]),
        "hist_gradient_boosting_climate": (
            HistGradientBoostingRegressor(**hgb_kwargs),
            feature_sets["climate"],
        ),
        "ridge_forecast": (ridge_forecast, feature_sets["forecast"]),
        "hist_gradient_boosting_forecast": (
            HistGradientBoostingRegressor(**hgb_kwargs),
            feature_sets["forecast"],
        ),
    }


def add_naive_predictions(df: pd.DataFrame) -> dict[str, str]:
    df["pred_previous_day_same_hour"] = df["lag_24"]
    df["pred_previous_week_same_hour"] = df["lag_168"]
    return {
        "previous_day_same_hour": "pred_previous_day_same_hour",
        "previous_week_same_hour": "pred_previous_week_same_hour",
    }


def train_and_predict(
    train_df: pd.DataFrame,
    eval_dfs: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, str]:
    prediction_columns: dict[str, str] = {}
    methods = set(config["models"]["baseline"].get("methods", []))
    model_specs = build_models(config)
    load_col = config["columns"]["load"]

    for model_name, (model, feature_cols) in model_specs.items():
        if model_name not in methods:
            continue
        check_columns(train_df, feature_cols + [load_col], model_name)
        train_clean = train_df.dropna(subset=feature_cols + [load_col])
        model.fit(train_clean[feature_cols], train_clean[load_col])

        pred_col = f"pred_{model_name}"
        prediction_columns[model_name] = pred_col
        for df in eval_dfs.values():
            check_columns(df, feature_cols, model_name)
            df[pred_col] = model.predict(df[feature_cols])

    return prediction_columns


def save_figures(
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    prediction_columns: dict[str, str],
    config: dict[str, Any],
) -> None:
    figures_dir = configured_path(config, "modeling_figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)
    datetime_col = config["columns"]["datetime"]
    load_col = config["columns"]["load"]

    plt.figure(figsize=(14, 5))
    plt.plot(test_df[datetime_col], test_df[load_col], label="actual", linewidth=1.0)
    for model_name, pred_col in prediction_columns.items():
        plt.plot(test_df[datetime_col], test_df[pred_col], label=model_name, alpha=0.75, linewidth=0.8)
    plt.title("Baseline Predictions on 2022 Test Period")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_predictions_test_timeseries.png", dpi=150)
    plt.close()

    test_overall = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["segment"] == "overall")
    ].sort_values("MAE")

    plt.figure(figsize=(8, 4))
    plt.bar(test_overall["model"], test_overall["MAE"])
    plt.title("Test MAE by Baseline Model")
    plt.xlabel("Model")
    plt.ylabel("MAE (MW)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_test_mae_by_model.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model_name, pred_col in prediction_columns.items():
        plt.scatter(test_df[load_col], test_df[pred_col], s=3, alpha=0.2, label=model_name)
    min_val = min(test_df[load_col].min(), *(test_df[col].min() for col in prediction_columns.values()))
    max_val = max(test_df[load_col].max(), *(test_df[col].max() for col in prediction_columns.values()))
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)
    plt.title("Actual vs Predicted Load")
    plt.xlabel("Actual Load (MW)")
    plt.ylabel("Predicted Load (MW)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_actual_vs_predicted_scatter.png", dpi=150)
    plt.close()


def train_baselines(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = configured_path(config, "model_dataset")
    metrics_path = configured_path(config, "baseline_metrics")
    predictions_path = configured_path(config, "baseline_predictions_test")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    datetime_col = config["columns"]["datetime"]
    load_col = config["columns"]["load"]
    df = pd.read_csv(dataset_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    splits = split_dataset(df, config, datetime_col)
    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "Train, validation, and test splits must all be non-empty. "
            f"Got train={len(train_df)}, validation={len(validation_df)}, test={len(test_df)}."
        )

    prediction_columns: dict[str, str] = {}
    for split_df in [validation_df, test_df]:
        prediction_columns.update(add_naive_predictions(split_df))

    model_prediction_columns = train_and_predict(
        train_df=train_df,
        eval_dfs={"validation": validation_df, "test": test_df},
        config=config,
    )
    prediction_columns.update(model_prediction_columns)

    metric_rows = []
    metric_rows.extend(evaluate_predictions("validation", validation_df, prediction_columns, config))
    metric_rows.extend(evaluate_predictions("test", test_df, prediction_columns, config))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)

    output_cols = [
        datetime_col,
        load_col,
        config["columns"]["temperature"],
        config["columns"]["relative_humidity"],
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "is_weekend",
        "split",
    ] + list(prediction_columns.values())
    test_df[output_cols].to_csv(predictions_path, index=False)

    save_figures(test_df, metrics_df, prediction_columns, config)

    print("Saved baseline metrics:", metrics_path)
    print("Saved test predictions:", predictions_path)
    print("Saved modeling figures:", configured_path(config, "modeling_figures_dir"))
    print("Split counts:")
    print(pd.Series({name: len(split_df) for name, split_df in splits.items()}).to_string())
    print("Test overall metrics:")
    print(
        metrics_df[
            (metrics_df["split"] == "test") & (metrics_df["segment"] == "overall")
        ][["model", "n", "MAE", "RMSE", "MAPE", "R2"]]
        .sort_values("MAE")
        .to_string(index=False)
    )
    return metrics_df, test_df


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_baselines(config)


if __name__ == "__main__":
    main()
