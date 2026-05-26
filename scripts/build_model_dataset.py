import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gridclimate.config import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a config-driven historical model dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def require_column(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Missing {label} column '{column}'. Available columns: {list(df.columns)}")


def parse_end_datetime(value: Any) -> pd.Timestamp:
    text = str(value)
    timestamp = pd.to_datetime(text)
    if len(text) == 10:
        return timestamp + pd.Timedelta(hours=23)
    return timestamp


def configured_path(config: dict[str, Any], key: str) -> Path:
    try:
        return resolve_project_path(config["paths"][key])
    except KeyError as exc:
        raise KeyError(f"Missing paths.{key} in config.") from exc


def load_load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = configured_path(config, "load_data")
    raw = pd.read_csv(path)
    raw_columns = config["raw_columns"]["load"]
    out_columns = config["columns"]

    time_col = raw_columns["datetime"]
    load_col = raw_columns["load"]
    require_column(raw, time_col, "load datetime")
    require_column(raw, load_col, "load value")

    df = raw[[time_col, load_col]].copy()
    df.columns = [out_columns["datetime"], out_columns["load"]]
    df[out_columns["datetime"]] = pd.to_datetime(df[out_columns["datetime"]], errors="coerce")
    df[out_columns["load"]] = pd.to_numeric(df[out_columns["load"]], errors="coerce")

    stats = {
        "path": str(path),
        "raw_rows": int(len(raw)),
        "invalid_datetime_rows": int(df[out_columns["datetime"]].isna().sum()),
        "missing_load_rows": int(df[out_columns["load"]].isna().sum()),
        "duplicate_datetime_rows": int(df.duplicated(out_columns["datetime"]).sum()),
    }

    df = df.dropna(subset=[out_columns["datetime"]])
    df = df.drop_duplicates(subset=[out_columns["datetime"]], keep="last")
    df = df.sort_values(out_columns["datetime"])
    return df, stats


def load_weather_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = configured_path(config, "weather_data")
    raw = pd.read_csv(path)
    raw_columns = config["raw_columns"]["weather"]
    out_columns = config["columns"]

    time_col = raw_columns["datetime"]
    temp_col = raw_columns["temperature"]
    rh_col = raw_columns["relative_humidity"]
    require_column(raw, time_col, "weather datetime")
    require_column(raw, temp_col, "temperature")
    require_column(raw, rh_col, "relative humidity")

    df = raw[[time_col, temp_col, rh_col]].copy()
    df.columns = [
        out_columns["datetime"],
        out_columns["temperature"],
        out_columns["relative_humidity"],
    ]
    df[out_columns["datetime"]] = pd.to_datetime(df[out_columns["datetime"]], errors="coerce")
    df[out_columns["temperature"]] = pd.to_numeric(df[out_columns["temperature"]], errors="coerce")
    df[out_columns["relative_humidity"]] = pd.to_numeric(
        df[out_columns["relative_humidity"]],
        errors="coerce",
    )

    stats = {
        "path": str(path),
        "raw_rows": int(len(raw)),
        "invalid_datetime_rows": int(df[out_columns["datetime"]].isna().sum()),
        "missing_temperature_rows": int(df[out_columns["temperature"]].isna().sum()),
        "missing_relative_humidity_rows": int(df[out_columns["relative_humidity"]].isna().sum()),
        "duplicate_datetime_rows": int(df.duplicated(out_columns["datetime"]).sum()),
    }

    df = df.dropna(subset=[out_columns["datetime"]])
    df = df.drop_duplicates(subset=[out_columns["datetime"]], keep="last")
    df = df.sort_values(out_columns["datetime"])
    return df, stats


def add_calendar_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[datetime_col].dt.year
    out["month"] = out[datetime_col].dt.month
    out["day"] = out[datetime_col].dt.day
    out["hour"] = out[datetime_col].dt.hour
    out["dayofweek"] = out[datetime_col].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, load_col: str, lag_hours: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lag_hours:
        out[f"lag_{lag}"] = out[load_col].shift(lag)
    return out


def add_split_labels(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    split = config["split"]
    train_years = set(split.get("train_years", []))
    validation_years = set(split.get("validation_years", []))
    test_years = set(split.get("test_years", []))

    out["split"] = "unassigned"
    out.loc[out["year"].isin(train_years), "split"] = "train"
    out.loc[out["year"].isin(validation_years), "split"] = "validation"
    out.loc[out["year"].isin(test_years), "split"] = "test"
    return out


def markdown_table(series: pd.Series) -> str:
    if series.empty:
        return "| column | value |\n| --- | --- |\n"
    rows = ["| column | value |", "| --- | --- |"]
    for key, value in series.items():
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows) + "\n"


def write_quality_report(
    report_path: Path,
    dataset_path: Path,
    load_stats: dict[str, Any],
    weather_stats: dict[str, Any],
    missing_before: pd.Series,
    missing_after: pd.Series,
    final_df: pd.DataFrame,
    datetime_col: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = final_df["split"].value_counts().sort_index()

    lines = [
        "# Data Quality Report",
        "",
        "## Output",
        "",
        f"- Model dataset: `{dataset_path}`",
        f"- Rows: {len(final_df)}",
        f"- Datetime min: {final_df[datetime_col].min()}",
        f"- Datetime max: {final_df[datetime_col].max()}",
        "",
        "## Raw Load Data",
        "",
        markdown_table(pd.Series(load_stats)),
        "## Raw Weather Data",
        "",
        markdown_table(pd.Series(weather_stats)),
        "## Missing Values Before Cleaning",
        "",
        markdown_table(missing_before),
        "## Missing Values After Cleaning",
        "",
        markdown_table(missing_after),
        "## Split Counts",
        "",
        markdown_table(split_counts),
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(df: pd.DataFrame) -> None:
    print("Row count:", len(df))
    print("Datetime min:", df["timestamp"].min())
    print("Datetime max:", df["timestamp"].max())
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Split counts:")
    print(df["split"].value_counts().sort_index().to_string())


def build_dataset(config: dict[str, Any]) -> pd.DataFrame:
    columns = config["columns"]
    datetime_col = columns["datetime"]
    load_col = columns["load"]
    temp_col = columns["temperature"]
    rh_col = columns["relative_humidity"]

    load_df, load_stats = load_load_data(config)
    weather_df, weather_stats = load_weather_data(config)

    start = pd.to_datetime(config["data"]["start_date"])
    end = parse_end_datetime(config["data"]["end_date"])
    frequency = config["data"].get("frequency", "h")
    hourly_index = pd.date_range(start=start, end=end, freq=frequency)

    df = pd.DataFrame({datetime_col: hourly_index})
    df = df.merge(load_df, on=datetime_col, how="left")
    df = df.merge(weather_df, on=datetime_col, how="left")

    missing_before = df[[load_col, temp_col, rh_col]].isna().sum()

    interpolate_limit = int(config["data"].get("weather_interpolate_limit_hours", 6))
    df[temp_col] = df[temp_col].interpolate(limit=interpolate_limit, limit_direction="both")
    df[rh_col] = df[rh_col].interpolate(limit=interpolate_limit, limit_direction="both")

    df = df.dropna(subset=[load_col]).copy()

    if config["features"].get("use_calendar_features", True):
        df = add_calendar_features(df, datetime_col)

    lag_hours = config["features"].get("lag_hours", [])
    if config["features"].get("use_lag_features", True):
        df = add_lag_features(df, load_col, lag_hours)

    rh_min = config.get("quality", {}).get("relative_humidity_min", 0)
    rh_max = config.get("quality", {}).get("relative_humidity_max", 100)
    df = df[(df[rh_col] >= rh_min) & (df[rh_col] <= rh_max)].copy()

    required_columns = [load_col]
    if config["features"].get("use_temperature", True):
        required_columns.append(temp_col)
    if config["features"].get("use_relative_humidity", True):
        required_columns.append(rh_col)
    if config["features"].get("use_lag_features", True):
        required_columns.extend([f"lag_{lag}" for lag in lag_hours])

    df = df.dropna(subset=required_columns).copy()
    df = add_split_labels(df, config)
    df = df.sort_values(datetime_col).reset_index(drop=True)

    dataset_path = configured_path(config, "model_dataset")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)

    report_path = configured_path(config, "data_quality_report")
    missing_after = df.isna().sum()
    write_quality_report(
        report_path=report_path,
        dataset_path=dataset_path,
        load_stats=load_stats,
        weather_stats=weather_stats,
        missing_before=missing_before,
        missing_after=missing_after,
        final_df=df,
        datetime_col=datetime_col,
    )

    print("Saved model dataset:", dataset_path)
    print("Saved data quality report:", report_path)
    print_summary(df)
    return df


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    build_dataset(config)


if __name__ == "__main__":
    main()
