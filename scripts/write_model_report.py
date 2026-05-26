import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridclimate.config import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a load forecasting model evaluation report."
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


def optional_path(relative_path: str) -> Path:
    return resolve_project_path(relative_path)


def display_path(path: Path) -> str:
    root = resolve_project_path(".").resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def fmt_number(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, digits: int = 2) -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    table = df.copy()
    if columns is not None:
        table = table[columns]
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(fmt_number(row[col], digits) for col in table.columns) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def data_coverage(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    datetime_col = config["columns"]["datetime"]
    load_col = config["columns"]["load"]
    temp_col = config["columns"]["temperature"]
    rh_col = config["columns"]["relative_humidity"]
    return {
        "rows": len(df),
        "datetime_min": df[datetime_col].min(),
        "datetime_max": df[datetime_col].max(),
        "missing_values": df[[datetime_col, load_col, temp_col, rh_col]].isna().sum().to_dict(),
        "split_counts": df["split"].value_counts().sort_index().to_dict()
        if "split" in df.columns
        else {},
    }


def eda_findings(tables_dir: Path) -> list[str]:
    findings: list[str] = []
    monthly_summary = read_csv_if_exists(tables_dir / "monthly_load_summary.csv")
    temp_response = read_csv_if_exists(tables_dir / "temperature_bin_response.csv")
    monthly_peak = read_csv_if_exists(tables_dir / "monthly_peak_load.csv")
    top50 = read_csv_if_exists(tables_dir / "top50_peak_load_events.csv")
    top5_temp = read_csv_if_exists(tables_dir / "top5_load_temperature_distribution.csv")
    diurnal = read_csv_if_exists(tables_dir / "weekday_weekend_diurnal_curve.csv")

    if monthly_summary is not None and not monthly_summary.empty:
        hottest = monthly_summary.loc[monthly_summary["temperature_mean_c"].idxmax()]
        highest_mean = monthly_summary.loc[monthly_summary["load_mean_mw"].idxmax()]
        findings.append(
            "The warmest monthly average temperature occurs in "
            f"{hottest['year_month']} ({fmt_number(hottest['temperature_mean_c'])} C), "
            f"while the highest monthly mean load occurs in {highest_mean['year_month']} "
            f"({fmt_number(highest_mean['load_mean_mw'])} MW)."
        )

    if temp_response is not None and not temp_response.empty:
        min_load_bin = temp_response.loc[temp_response["load_mw"].idxmin()]
        max_load_bin = temp_response.loc[temp_response["load_mw"].idxmax()]
        findings.append(
            "The temperature-load response is U-shaped: average load is lowest near "
            f"{fmt_number(min_load_bin['temperature_bin_mid'])} C "
            f"({fmt_number(min_load_bin['load_mw'])} MW) and highest near "
            f"{fmt_number(max_load_bin['temperature_bin_mid'])} C "
            f"({fmt_number(max_load_bin['load_mw'])} MW)."
        )

    if monthly_peak is not None and not monthly_peak.empty:
        peak_month = monthly_peak.loc[monthly_peak["peak_load_mw"].idxmax()]
        findings.append(
            f"The maximum monthly peak load is {fmt_number(peak_month['peak_load_mw'])} MW "
            f"in {peak_month['year_month']}."
        )

    if top50 is not None and not top50.empty:
        hot_share = (top50["temperature_c"] >= 30).mean() * 100
        winter_share = top50["month"].isin([12, 1, 2]).mean() * 100
        findings.append(
            f"Among the top 50 peak-load events, {fmt_number(hot_share, 1)}% occur at "
            f"temperature >= 30 C and {fmt_number(winter_share, 1)}% occur in DJF months."
        )

    if top5_temp is not None and not top5_temp.empty:
        dominant_bin = top5_temp.loc[top5_temp["hour_count"].idxmax()]
        findings.append(
            "The most common temperature bin among top 5% load hours is "
            f"{dominant_bin['temperature_bin_label']} "
            f"({fmt_number(dominant_bin['share_of_top_load_hours'] * 100, 1)}% of top-load hours)."
        )

    if diurnal is not None and not diurnal.empty:
        peak_hours = (
            diurnal.sort_values("load_mw", ascending=False)
            .groupby("day_type")
            .head(1)
            .sort_values("day_type")
        )
        if not peak_hours.empty:
            text = ", ".join(
                f"{row['day_type']} peak at hour {int(row['hour'])}"
                for _, row in peak_hours.iterrows()
            )
            findings.append(f"The diurnal summaries show {text}.")

    return findings


def test_overall_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics[(metrics["split"] == "test") & (metrics["segment"] == "overall")]
        .sort_values("MAE")
        .reset_index(drop=True)
    )


def segment_metrics(metrics: pd.DataFrame, segment: str) -> pd.DataFrame:
    return (
        metrics[(metrics["split"] == "test") & (metrics["segment"] == segment)]
        .sort_values("MAE")
        .reset_index(drop=True)
    )


def climate_sufficiency_text(metrics: pd.DataFrame) -> str:
    test_overall = test_overall_metrics(metrics)
    climate = test_overall[test_overall["model"].str.endswith("_climate")]
    forecast = test_overall[test_overall["model"].str.endswith("_forecast")]
    if climate.empty or forecast.empty:
        return (
            "The current metric table does not contain both climate-only and forecast-feature "
            "models, so climate-feature sufficiency cannot be assessed from this run."
        )

    best_climate = climate.iloc[0]
    best_forecast = forecast.iloc[0]
    ratio = best_climate["MAE"] / best_forecast["MAE"]
    return (
        "Climate-only features capture a meaningful part of the load-temperature relationship, "
        f"but they are not sufficient for high-accuracy historical forecasting in the current setup. "
        f"The best climate-only model is `{best_climate['model']}` with test MAE "
        f"{fmt_number(best_climate['MAE'])} MW, while the best forecast-feature model is "
        f"`{best_forecast['model']}` with test MAE {fmt_number(best_forecast['MAE'])} MW "
        f"({fmt_number(ratio, 1)}x lower MAE for the forecast-feature model). "
        "For future climate scenario simulation, climate-only models are still useful because "
        "future observed lagged load is unavailable, but they should be treated as a scenario "
        "response model rather than a final operational forecaster."
    )


def load_lstm_metrics() -> pd.DataFrame | None:
    candidates = [
        optional_path("outputs/model_results/lstm_metrics.csv"),
        optional_path("outputs/tables/lstm_metrics.csv"),
        optional_path("outputs/tables/model_metrics_comparison.csv"),
    ]
    for path in candidates:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            df = df.copy()
            df["source_file"] = display_path(path)
            return df
    return None


def prediction_summary(predictions: pd.DataFrame | None, load_col: str) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    pred_cols = [col for col in predictions.columns if col.startswith("pred_")]
    rows = []
    for col in pred_cols:
        residual = predictions[col] - predictions[load_col]
        rows.append(
            {
                "model": col.replace("pred_", ""),
                "mean_residual_mw": residual.mean(),
                "median_abs_error_mw": residual.abs().median(),
                "max_abs_error_mw": residual.abs().max(),
            }
        )
    return pd.DataFrame(rows).sort_values("median_abs_error_mw")


def write_report(config: dict[str, Any]) -> Path:
    dataset_path = configured_path(config, "model_dataset")
    tables_dir = configured_path(config, "eda_tables_dir")
    baseline_metrics_path = configured_path(config, "baseline_metrics")
    baseline_predictions_path = configured_path(config, "baseline_predictions_test")
    report_path = configured_path(config, "model_evaluation_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    columns = config["columns"]
    datetime_col = columns["datetime"]
    load_col = columns["load"]

    dataset = pd.read_csv(dataset_path, parse_dates=[datetime_col])
    metrics = pd.read_csv(baseline_metrics_path)
    predictions = read_csv_if_exists(baseline_predictions_path)
    lstm_metrics = load_lstm_metrics()

    coverage = data_coverage(dataset, config)
    findings = eda_findings(tables_dir)
    overall = test_overall_metrics(metrics)
    hot30 = segment_metrics(metrics, "hot_hours_temp_ge_30")
    hot33 = segment_metrics(metrics, "hot_hours_temp_ge_33")
    peak = segment_metrics(metrics, "peak_top_5_percent_load")
    pred_summary = prediction_summary(predictions, load_col)

    feature_sets = config["models"]["baseline"].get("feature_sets", {})
    lines = [
        "# Load Forecasting Model Evaluation",
        "",
        "## Data Coverage",
        "",
        f"- Model dataset: `{display_path(dataset_path)}`",
        f"- Rows: {fmt_number(coverage['rows'], 0)}",
        f"- Datetime range: {coverage['datetime_min']} to {coverage['datetime_max']}",
        f"- Missing values in core columns: `{coverage['missing_values']}`",
        f"- Split counts: `{coverage['split_counts']}`",
        "",
        "## Feature Construction",
        "",
        "- Target: hourly electricity load (`load_mw`).",
        "- Climate features: "
        + ", ".join(f"`{feature}`" for feature in feature_sets.get("climate", []))
        + ".",
        "- Forecast features: "
        + ", ".join(f"`{feature}`" for feature in feature_sets.get("forecast", []))
        + ".",
        "- Lag features are generated from historical load: "
        + ", ".join(f"`lag_{lag}`" for lag in config["features"].get("lag_hours", []))
        + ".",
        "",
        "## Train/Validation/Test Split",
        "",
        f"- Train: {config['split']['train_start']} to {config['split']['train_end']}.",
        f"- Validation: {config['split']['validation_start']} to {config['split']['validation_end']}.",
        f"- Test: {config['split']['test_start']} to {config['split']['test_end']}.",
        "",
        "## Key EDA Findings",
        "",
    ]

    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- EDA tables were not found, so no automated EDA findings were generated.")

    lines.extend(
        [
            "",
            "## Baseline Performance",
            "",
            "Test-set overall metrics:",
            "",
            markdown_table(overall[["model", "n", "MAE", "RMSE", "MAPE", "R2"]], digits=3),
            "",
            "Prediction residual summary on the 2022 test set:",
            "",
            markdown_table(pred_summary, digits=3),
            "",
            "## LSTM Performance",
            "",
        ]
    )

    if lstm_metrics is not None:
        lines.extend(
            [
                f"LSTM metrics were found in `{lstm_metrics['source_file'].iloc[0]}`.",
                "",
                markdown_table(lstm_metrics.drop(columns=["source_file"], errors="ignore"), digits=3),
            ]
        )
    else:
        lines.append(
            "No LSTM metrics file was found. This is acceptable for the current config-driven "
            "baseline stage; the prototype LSTM workflow should be refactored into the same "
            "configuration system before it is treated as a reproducible thesis result."
        )

    lines.extend(
        [
            "",
            "## Hot-Hour and Peak-Load Performance",
            "",
            "Hot hours, temperature >= 30 C:",
            "",
            markdown_table(hot30[["model", "n", "MAE", "RMSE", "MAPE", "R2"]], digits=3),
            "",
            "Hot hours, temperature >= 33 C:",
            "",
            markdown_table(hot33[["model", "n", "MAE", "RMSE", "MAPE", "R2"]], digits=3),
            "",
            "Top 5% load hours:",
            "",
            markdown_table(peak[["model", "n", "MAE", "RMSE", "MAPE", "R2"]], digits=3),
            "",
            "## Are Climate-Only Features Sufficient for Future Scenario Simulation?",
            "",
            climate_sufficiency_text(metrics),
            "",
            "## Limitations and Next Steps",
            "",
            "- Raw data are not distributed in the repository; replication requires users to obtain comparable TEPCO/OCCTO-type load data and JMA-type weather observations.",
            "- The current baseline models use historical observed weather. They do not yet include CMIP6 bias correction, downscaling, PV/wind generation, or storage optimization.",
            "- Forecast-feature models use lagged observed load, which improves accuracy but is not directly available for long-range future climate scenarios without a recursive or scenario-consistent demand simulation design.",
            "- Climate-only baseline performance should be improved with richer future-available predictors such as holiday flags, humidity-derived discomfort indices, nonlinear temperature terms, cooling/heating degree hours, and long-term socioeconomic demand assumptions.",
            "- Next steps: refactor LSTM training into the config-driven pipeline, add scenario-ready climate-only demand models, and then connect future demand to renewable supply and storage capacity estimation.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved model evaluation report:", report_path)
    print("Report rows used:")
    print(f"- dataset rows: {len(dataset)}")
    print(f"- baseline metric rows: {len(metrics)}")
    print(f"- prediction rows: {0 if predictions is None else len(predictions)}")
    print(f"- LSTM metrics available: {lstm_metrics is not None}")
    return report_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    write_report(config)


if __name__ == "__main__":
    main()
