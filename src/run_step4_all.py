import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED, FIGURES

def save_line_plot(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PROCESSED / "merged_dataset_hourly.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    save_line_plot(
        df["timestamp"],
        df["load_mw"],
        "TEPCO Load Time Series",
        "Time",
        "Load (MW)",
        FIGURES / "load_timeseries.png"
    )

    save_line_plot(
        df["timestamp"],
        df["temperature_c"],
        "Tokyo Temperature Time Series",
        "Time",
        "Temperature (°C)",
        FIGURES / "temperature_timeseries.png"
    )

    save_line_plot(
        df["timestamp"],
        df["rh_percent"],
        "Tokyo Relative Humidity Time Series",
        "Time",
        "RH (%)",
        FIGURES / "rh_timeseries.png"
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(df["temperature_c"], df["load_mw"], s=2)
    plt.title("Load vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "load_vs_temperature.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["rh_percent"], df["load_mw"], s=2)
    plt.title("Load vs Relative Humidity")
    plt.xlabel("RH (%)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "load_vs_rh.png", dpi=150)
    plt.close()

    by_hour = df.groupby("hour", as_index=False)["load_mw"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(by_hour["hour"], by_hour["load_mw"])
    plt.title("Average Load by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.xticks(range(0, 24, 1))
    plt.tight_layout()
    plt.savefig(FIGURES / "mean_load_by_hour.png", dpi=150)
    plt.close()

    by_month = df.groupby("month", as_index=False)["load_mw"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(by_month["month"], by_month["load_mw"])
    plt.title("Average Load by Month")
    plt.xlabel("Month")
    plt.ylabel("Load (MW)")
    plt.xticks(range(1, 13, 1))
    plt.tight_layout()
    plt.savefig(FIGURES / "mean_load_by_month.png", dpi=150)
    plt.close()

    preview = df[["load_mw", "temperature_c", "rh_percent"]].head(500).isna()

    plt.figure(figsize=(8, 4))
    plt.imshow(preview.T, aspect="auto")
    plt.yticks([0, 1, 2], ["load_mw", "temperature_c", "rh_percent"])
    plt.title("Missing Data Preview (First 500 Rows)")
    plt.xlabel("Row Index")
    plt.tight_layout()
    plt.savefig(FIGURES / "missing_matrix_preview.png", dpi=150)
    plt.close()

    print("Exploration figures saved to:", FIGURES)

if __name__ == "__main__":
    main()