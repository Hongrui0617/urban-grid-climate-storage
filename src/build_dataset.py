import json
from pathlib import Path

import pandas as pd

from config import (
    LOAD_FILE,
    WEATHER_FILE,
    DATA_PROCESSED,
    MERGED_DATASET_CSV,
    MERGED_DATASET_PARQUET,
    LAGS,
)

def detect_time_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not candidates:
        raise ValueError("No datetime-like column found.")
    return candidates[0]

def load_tepco_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = "datetime_jst" if "datetime_jst" in df.columns else detect_time_column(df)

    if "load_MW" in df.columns:
        load_col = "load_MW"
    else:
        candidates = [c for c in df.columns if "load" in c.lower() or "demand" in c.lower()]
        if not candidates:
            raise ValueError("No load column found in load data.")
        load_col = candidates[0]

    out = df[[time_col, load_col]].copy()
    out.columns = ["timestamp", "load_mw"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")
    out = out.dropna(subset=["timestamp", "load_mw"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"])
    return out

def load_weather_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = "datetime" if "datetime" in df.columns else detect_time_column(df)

    temp_col = "temp_c" if "temp_c" in df.columns else None
    rh_col = "rh_percent" if "rh_percent" in df.columns else None

    if temp_col is None:
        candidates = [c for c in df.columns if "temp" in c.lower()]
        if not candidates:
            raise ValueError("No temperature column found.")
        temp_col = candidates[0]

    if rh_col is None:
        candidates = [c for c in df.columns if "rh" in c.lower() or "humidity" in c.lower()]
        if not candidates:
            raise ValueError("No RH column found.")
        rh_col = candidates[0]

    out = df[[time_col, temp_col, rh_col]].copy()
    out.columns = ["timestamp", "temperature_c", "rh_percent"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="coerce")
    out["rh_percent"] = pd.to_numeric(out["rh_percent"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"])
    return out

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df["load_mw"].shift(lag)
    return df

def build_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="H")
    out = df.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()
    return out

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    load_df = load_tepco_data(LOAD_FILE)
    weather_df = load_weather_data(WEATHER_FILE)

    df = pd.merge(load_df, weather_df, on="timestamp", how="outer").sort_values("timestamp")
    df = build_hourly_index(df)

    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_ratio": df.isna().mean().values,
    })
    missing_report.to_csv(DATA_PROCESSED / "missing_report.csv", index=False)

    # interpolate weather only
    df["temperature_c"] = df["temperature_c"].interpolate(limit=6)
    df["rh_percent"] = df["rh_percent"].interpolate(limit=6)

    # keep only rows with valid target
    df = df.dropna(subset=["load_mw"]).copy()

    df = add_calendar_features(df)
    df = add_lag_features(df)

    df = df.dropna().copy()
    df = df[(df["rh_percent"] >= 0) & (df["rh_percent"] <= 100)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(MERGED_DATASET_CSV, index=False)
    df.to_parquet(MERGED_DATASET_PARQUET, index=False)

    summary = {
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "columns": list(df.columns),
    }
    with open(DATA_PROCESSED / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", MERGED_DATASET_CSV)
    print("Saved:", MERGED_DATASET_PARQUET)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()