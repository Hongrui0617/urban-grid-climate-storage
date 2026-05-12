import copy
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import (
    MERGED_DATASET_PARQUET,
    FEATURE_COLS,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    MODELS,
    TABLES,
    SEQ_LEN,
    BATCH_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    DEVICE,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * len(yb)
            n += len(yb)
    return total_loss / n

def main():
    set_seed(RANDOM_SEED)

    if DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MERGED_DATASET_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df, val_df, test_df = train_val_test_split(df)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[FEATURE_COLS] = x_scaler.fit_transform(train_df[FEATURE_COLS])
    val_df_scaled[FEATURE_COLS] = x_scaler.transform(val_df[FEATURE_COLS])
    test_df_scaled[FEATURE_COLS] = x_scaler.transform(test_df[FEATURE_COLS])

    train_df_scaled[[TARGET_COL]] = y_scaler.fit_transform(train_df[[TARGET_COL]])
    val_df_scaled[[TARGET_COL]] = y_scaler.transform(val_df[[TARGET_COL]])
    test_df_scaled[[TARGET_COL]] = y_scaler.transform(test_df[[TARGET_COL]])

    joblib.dump(x_scaler, MODELS / "lstm_x_scaler.pkl")
    joblib.dump(y_scaler, MODELS / "lstm_y_scaler.pkl")

    train_ds = SequenceDataset(train_df_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    val_ds = SequenceDataset(val_df_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    best_model = None
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            n += len(yb)

        train_loss = running_loss / n
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch:02d} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    torch.save(best_model, MODELS / "lstm_best.pt")

    with open(TABLES / "lstm_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Saved best model to:", MODELS / "lstm_best.pt")

if __name__ == "__main__":
    main()