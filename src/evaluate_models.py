import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    MERGED_DATASET_PARQUET,
    FEATURE_COLS,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    MODELS,
    TABLES,
    FIGURES,
    SEQ_LEN,
    BATCH_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    DEVICE,
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def peak_mae(y_true, y_pred, q=0.99):
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])

def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Top1pct_MAE": peak_mae(y_true, y_pred, q=0.99),
    }

def train_val_test_split(df):
    train = df[df["timestamp"] <= TRAIN_END].copy()
    val = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)].copy()
    test = df[df["timestamp"] > VAL_END].copy()
    return train, val, test

class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

def save_timeseries_plot(df, actual_col, pred_col, title, save_path):
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df[actual_col], label="Actual")
    plt.plot(df["timestamp"], df[pred_col], label=pred_col)
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_scatter_plot(actual, pred, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, s=2)
    min_val = min(np.min(actual), np.min(pred))
    max_val = max(np.max(actual), np.max(pred))
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Load (MW)")
    plt.ylabel("Predicted Load (MW)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    df = pd.read_parquet(MERGED_DATASET_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    _, _, test_df = train_val_test_split(df)

    baseline_preds = pd.read_csv(TABLES / "baseline_predictions_test.csv")
    baseline_preds["timestamp"] = pd.to_datetime(baseline_preds["timestamp"])

    x_scaler = joblib.load(MODELS / "lstm_x_scaler.pkl")
    y_scaler = joblib.load(MODELS / "lstm_y_scaler.pkl")

    test_df_scaled = test_df.copy()
    test_df_scaled[FEATURE_COLS] = x_scaler.transform(test_df[FEATURE_COLS])
    test_df_scaled[[TARGET_COL]] = y_scaler.transform(test_df[[TARGET_COL]])

    test_ds = SequenceDataset(test_df_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(MODELS / "lstm_best.pt", map_location=device))
    model.eval()

    preds_scaled = []
    y_scaled = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.extend(pred.tolist())
            y_scaled.extend(yb.numpy().tolist())

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    y_scaled = np.array(y_scaled).reshape(-1, 1)

    y_pred = y_scaler.inverse_transform(preds_scaled).ravel()
    y_true = y_scaler.inverse_transform(y_scaled).ravel()

    aligned_test = test_df.iloc[SEQ_LEN:].copy().reset_index(drop=True)

    lstm_df = pd.DataFrame({
        "timestamp": pd.to_datetime(aligned_test["timestamp"].values),
        "load_mw": y_true,
        "y_pred_lstm": y_pred,
        "temperature_c": aligned_test["temperature_c"].values,
        "rh_percent": aligned_test["rh_percent"].values,
    })

    lstm_df["residual_lstm"] = lstm_df["y_pred_lstm"] - lstm_df["load_mw"]
    lstm_df["abs_error_lstm"] = np.abs(lstm_df["residual_lstm"])
    lstm_df.to_csv(TABLES / "lstm_predictions_test.csv", index=False)

    compare_df = pd.merge(
       baseline_preds,
       lstm_df[["timestamp", "y_pred_lstm", "residual_lstm", "abs_error_lstm", "temperature_c", "rh_percent"]],
       on="timestamp",
       how="inner"
    ).sort_values("timestamp").reset_index(drop=True)

    compare_df["residual_rf"] = compare_df["pred_random_forest"] - compare_df["load_mw"]
    compare_df["abs_error_rf"] = np.abs(compare_df["residual_rf"])

    compare_df.to_csv(TABLES / "model_comparison_test_aligned.csv", index=False)

    metrics = []
    metrics.append(evaluate("random_forest_aligned", compare_df["load_mw"], compare_df["pred_random_forest"]))
    metrics.append(evaluate("lstm", compare_df["load_mw"], compare_df["y_pred_lstm"]))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(TABLES / "model_metrics_comparison.csv", index=False)
    metrics_df.to_csv(TABLES / "lstm_metrics.csv", index=False)

    # 1. full combined comparison plot (keep as supplementary)
    plt.figure(figsize=(14, 5))
    plt.plot(compare_df["timestamp"], compare_df["load_mw"], label="Actual")
    plt.plot(compare_df["timestamp"], compare_df["pred_random_forest"], label="RandomForest", alpha=0.8)
    plt.plot(compare_df["timestamp"], compare_df["y_pred_lstm"], label="LSTM", alpha=0.8)
    plt.legend()
    plt.title("Test Period Load Prediction Comparison")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "test_year_prediction_comparison.png", dpi=150)
    plt.close()

    # 2. separate timeseries plots
    save_timeseries_plot(
        compare_df,
        actual_col="load_mw",
        pred_col="pred_random_forest",
        title="Actual vs RandomForest (Test Period)",
        save_path=FIGURES / "test_actual_vs_rf.png"
    )

    save_timeseries_plot(
        compare_df,
        actual_col="load_mw",
        pred_col="y_pred_lstm",
        title="Actual vs LSTM (Test Period)",
        save_path=FIGURES / "test_actual_vs_lstm.png"
    )

    # 3. scatter plots
    save_scatter_plot(
        compare_df["load_mw"].values,
        compare_df["pred_random_forest"].values,
        "Actual vs RandomForest",
        FIGURES / "scatter_actual_vs_rf.png"
    )

    save_scatter_plot(
        compare_df["load_mw"].values,
        compare_df["y_pred_lstm"].values,
        "Actual vs LSTM",
        FIGURES / "scatter_actual_vs_lstm.png"
    )

    # 4. residual vs temperature bins for LSTM
    compare_df["temp_bin"] = pd.cut(compare_df["temperature_c"], bins=15)
    binned = compare_df.groupby("temp_bin", observed=False)["residual_lstm"].mean().reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(binned)), binned["residual_lstm"])
    plt.xticks(range(len(binned)), [str(x) for x in binned["temp_bin"]], rotation=90)
    plt.title("Mean Residual by Temperature Bin (LSTM)")
    plt.xlabel("Temperature Bin")
    plt.ylabel("Mean Residual (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_vs_temp_bins.png", dpi=150)
    plt.close()

    # 5. absolute error boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot([
        compare_df["abs_error_rf"].values,
        compare_df["abs_error_lstm"].values
    ], labels=["RandomForest", "LSTM"])
    plt.title("Absolute Error Distribution: RF vs LSTM")
    plt.ylabel("Absolute Error (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "abs_error_boxplot_rf_vs_lstm.png", dpi=150)
    plt.close()

    # 6. top 20 peak hours point comparison
    top20 = compare_df.nlargest(20, "load_mw").copy().sort_values("timestamp")
    top20.to_csv(TABLES / "top20_peak_hours_lstm.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(top20)), top20["load_mw"].values, marker="o", label="Actual")
    plt.plot(range(len(top20)), top20["pred_random_forest"].values, marker="o", label="RandomForest")
    plt.plot(range(len(top20)), top20["y_pred_lstm"].values, marker="o", label="LSTM")
    plt.xticks(
        range(len(top20)),
        top20["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        rotation=90
    )
    plt.title("Top 20 Peak Hours: Actual vs RF vs LSTM")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "top20_peak_hours_point_compare.png", dpi=150)
    plt.close()

    # 7. old-style peak absolute error bar chart (keep if useful)
    top20_err = top20.copy()
    top20_err["abs_error_lstm"] = np.abs(top20_err["y_pred_lstm"] - top20_err["load_mw"])

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top20_err)), top20_err["abs_error_lstm"])
    plt.xticks(
        range(len(top20_err)),
        top20_err["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        rotation=90
    )
    plt.title("Top 20 Peak Hours Absolute Error (LSTM)")
    plt.xlabel("Timestamp")
    plt.ylabel("Absolute Error (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES / "top20_peak_hours_error.png", dpi=150)
    plt.close()

    print("Evaluation completed.")
    print("\nSaved figures:")
    print(FIGURES / "test_year_prediction_comparison.png")
    print(FIGURES / "test_actual_vs_rf.png")
    print(FIGURES / "test_actual_vs_lstm.png")
    print(FIGURES / "scatter_actual_vs_rf.png")
    print(FIGURES / "scatter_actual_vs_lstm.png")
    print(FIGURES / "residual_vs_temp_bins.png")
    print(FIGURES / "abs_error_boxplot_rf_vs_lstm.png")
    print(FIGURES / "top20_peak_hours_point_compare.png")
    print(FIGURES / "top20_peak_hours_error.png")

    print("\nMetrics:")
    print(metrics_df)

if __name__ == "__main__":
    main()