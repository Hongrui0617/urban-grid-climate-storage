import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "urban_grid_climate_storage_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridclimate.config import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA for the config-driven historical model dataset."
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


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_line_plot(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, linewidth=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def add_temperature_bins(df: pd.DataFrame, temp_col: str, bin_width: float) -> pd.DataFrame:
    out = df.copy()
    min_edge = np.floor(out[temp_col].min() / bin_width) * bin_width
    max_edge = np.ceil(out[temp_col].max() / bin_width) * bin_width + bin_width
    bins = np.arange(min_edge, max_edge + bin_width, bin_width)
    out["temperature_bin"] = pd.cut(out[temp_col], bins=bins, include_lowest=True)
    out["temperature_bin_mid"] = out["temperature_bin"].apply(lambda interval: interval.mid)
    out["temperature_bin_label"] = out["temperature_bin"].astype(str)
    return out


def save_heatmap(
    values: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    dpi: int,
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(10, 5))
    image = plt.imshow(values.values, aspect="auto", cmap=cmap)
    plt.colorbar(image, label="Load (MW)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(values.columns)), values.columns)
    plt.yticks(range(len(values.index)), values.index)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def print_summary(df: pd.DataFrame, datetime_col: str) -> None:
    print("Row count:", len(df))
    print("Datetime min:", df[datetime_col].min())
    print("Datetime max:", df[datetime_col].max())
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Split counts:")
    if "split" in df.columns:
        print(df["split"].value_counts().sort_index().to_string())
    else:
        print("split column not found")


def run_eda(config: dict[str, Any]) -> pd.DataFrame:
    columns = config["columns"]
    datetime_col = columns["datetime"]
    load_col = columns["load"]
    temp_col = columns["temperature"]
    rh_col = columns["relative_humidity"]

    dataset_path = configured_path(config, "model_dataset")
    figures_dir = configured_path(config, "eda_figures_dir")
    tables_dir = configured_path(config, "eda_tables_dir")
    ensure_dirs(figures_dir, tables_dir)

    eda_config = config.get("eda", {})
    dpi = int(eda_config.get("figure_dpi", 150))
    temp_bin_width = float(eda_config.get("temperature_bin_width_c", 2))
    top_load_percentile = float(eda_config.get("top_load_percentile", 0.95))
    top_peak_events = int(eda_config.get("top_peak_events", 50))

    df = pd.read_csv(dataset_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df["year_month"] = df[datetime_col].dt.to_period("M").astype(str)
    df = add_temperature_bins(df, temp_col, temp_bin_width)

    save_line_plot(
        df[datetime_col],
        df[load_col],
        "Load Time Series",
        "Datetime",
        "Load (MW)",
        figures_dir / "load_time_series.png",
        dpi,
    )

    month_hour = df.pivot_table(
        index="month",
        columns="hour",
        values=load_col,
        aggfunc="mean",
        observed=False,
    )
    month_hour.to_csv(tables_dir / "month_hour_load.csv")
    save_heatmap(
        month_hour,
        "Mean Load by Month and Hour",
        "Hour",
        "Month",
        figures_dir / "month_hour_heatmap.png",
        dpi,
    )

    diurnal = (
        df.groupby(["is_weekend", "hour"], observed=False)[load_col]
        .mean()
        .reset_index()
    )
    diurnal["day_type"] = diurnal["is_weekend"].map({0: "weekday", 1: "weekend"})
    diurnal.to_csv(tables_dir / "weekday_weekend_diurnal_curve.csv", index=False)
    plt.figure(figsize=(8, 4))
    for day_type, group in diurnal.groupby("day_type"):
        plt.plot(group["hour"], group[load_col], marker="o", label=day_type)
    plt.title("Weekday and Weekend Diurnal Load Curve")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "weekday_weekend_diurnal_curve.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(df[temp_col], df[load_col], s=3, alpha=0.25)
    plt.title("Load vs Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(figures_dir / "temperature_load_scatter.png", dpi=dpi)
    plt.close()

    temperature_bin_summary = (
        df.groupby("temperature_bin", observed=False)
        .agg(
            count=(load_col, "size"),
            temperature_mean_c=(temp_col, "mean"),
            load_mean_mw=(load_col, "mean"),
            load_median_mw=(load_col, "median"),
            load_min_mw=(load_col, "min"),
            load_max_mw=(load_col, "max"),
        )
        .reset_index()
    )
    temperature_bin_summary["temperature_bin"] = temperature_bin_summary[
        "temperature_bin"
    ].astype(str)
    temperature_bin_summary.to_csv(tables_dir / "temperature_bin_summary.csv", index=False)

    response = (
        df.groupby(["temperature_bin_mid"], observed=False)[load_col]
        .mean()
        .reset_index()
        .dropna()
    )
    response["temperature_bin_mid"] = response["temperature_bin_mid"].astype(float)
    response = response.sort_values("temperature_bin_mid")
    response.to_csv(tables_dir / "temperature_bin_response.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.plot(response["temperature_bin_mid"], response[load_col], marker="o")
    plt.title("Temperature-Bin Load Response")
    plt.xlabel("Temperature Bin Midpoint (C)")
    plt.ylabel("Mean Load (MW)")
    plt.tight_layout()
    plt.savefig(figures_dir / "temperature_bin_response.png", dpi=dpi)
    plt.close()

    hourly_response = df.pivot_table(
        index="temperature_bin",
        columns="hour",
        values=load_col,
        aggfunc="mean",
        observed=False,
    )
    hourly_response.index = hourly_response.index.astype(str)
    hourly_response.to_csv(tables_dir / "hourly_temperature_response.csv")
    save_heatmap(
        hourly_response,
        "Hourly Temperature Response",
        "Hour",
        "Temperature Bin",
        figures_dir / "hourly_temperature_response.png",
        dpi,
        cmap="magma",
    )

    monthly_peak = (
        df.groupby("year_month", observed=False)
        .agg(peak_load_mw=(load_col, "max"))
        .reset_index()
    )
    monthly_peak.to_csv(tables_dir / "monthly_peak_load.csv", index=False)
    plt.figure(figsize=(12, 4))
    plt.plot(monthly_peak["year_month"], monthly_peak["peak_load_mw"], marker="o")
    plt.title("Monthly Peak Load")
    plt.xlabel("Month")
    plt.ylabel("Peak Load (MW)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_peak_load.png", dpi=dpi)
    plt.close()

    threshold = df[load_col].quantile(top_load_percentile)
    top_load = df[df[load_col] >= threshold].copy()
    top_temperature_distribution = (
        top_load.groupby("temperature_bin", observed=False)
        .agg(
            hour_count=(load_col, "size"),
            load_mean_mw=(load_col, "mean"),
            temperature_mean_c=(temp_col, "mean"),
        )
        .reset_index()
    )
    top_temperature_distribution["temperature_bin_mid"] = top_temperature_distribution[
        "temperature_bin"
    ].apply(lambda interval: interval.mid)
    top_temperature_distribution["temperature_bin_label"] = top_temperature_distribution[
        "temperature_bin"
    ].astype(str)
    top_temperature_distribution = top_temperature_distribution[
        top_temperature_distribution["hour_count"] > 0
    ].copy()
    top_temperature_distribution["share_of_top_load_hours"] = (
        top_temperature_distribution["hour_count"] / len(top_load)
    )
    top_temperature_distribution = top_temperature_distribution.sort_values(
        "temperature_bin_mid"
    )
    top_temperature_distribution = top_temperature_distribution[
        [
            "temperature_bin_label",
            "temperature_bin_mid",
            "hour_count",
            "load_mean_mw",
            "temperature_mean_c",
            "share_of_top_load_hours",
        ]
    ]
    top_temperature_distribution.to_csv(
        tables_dir / "top5_load_temperature_distribution.csv",
        index=False,
    )
    plt.figure(figsize=(8, 4))
    plt.hist(df[temp_col], bins=30, alpha=0.4, label="All hours")
    plt.hist(top_load[temp_col], bins=30, alpha=0.7, label="Top 5% load hours")
    plt.title("Temperature Distribution During Top 5% Load Hours")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Hour Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "top5_load_temperature_distribution.png", dpi=dpi)
    plt.close()

    top50_columns = [
        datetime_col,
        load_col,
        temp_col,
        rh_col,
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "is_weekend",
        "split",
    ]
    top50 = df.nlargest(top_peak_events, load_col)[top50_columns]
    top50.to_csv(tables_dir / "top50_peak_load_events.csv", index=False)

    monthly_load_summary = (
        df.groupby("year_month", observed=False)
        .agg(
            row_count=(load_col, "size"),
            load_mean_mw=(load_col, "mean"),
            load_median_mw=(load_col, "median"),
            load_min_mw=(load_col, "min"),
            load_max_mw=(load_col, "max"),
            load_std_mw=(load_col, "std"),
            temperature_mean_c=(temp_col, "mean"),
        )
        .reset_index()
    )
    monthly_load_summary.to_csv(tables_dir / "monthly_load_summary.csv", index=False)

    print("Saved EDA figures:", figures_dir)
    print("Saved EDA tables:", tables_dir)
    print_summary(df.drop(columns=["temperature_bin", "temperature_bin_mid", "temperature_bin_label"]), datetime_col)
    return df


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_eda(config)


if __name__ == "__main__":
    main()
