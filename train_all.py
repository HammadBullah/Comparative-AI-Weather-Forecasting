"""
train_all.py
------------
Trains SARIMA, Random Forest, and LSTM for each of the 5 climate zones.
Run once after fetch_training_data.py.

Usage:
    python train_all.py

Output: models/<zone>_<model>.pkl / .pt
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("models", exist_ok=True)

ZONES    = ["tropical", "arid", "temperate", "continental", "polar"]
VARIABLE = "TG"
WINDOW   = 30


# ── PyTorch LSTM ──────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Train SARIMA ──────────────────────────────────────────────────────────────
def train_sarima(series: pd.Series, zone: str):
    print(f"  Training SARIMA for {zone}...")
    train = series.iloc[:int(len(series) * 0.8)]
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    joblib.dump(model, f"models/{zone}_sarima.pkl")
    print(f"    ✅ models/{zone}_sarima.pkl saved")


# ── Train Random Forest ───────────────────────────────────────────────────────
def train_rf(series: pd.Series, zone: str):
    print(f"  Training RF for {zone}...")
    data = series.values
    X, y = [], []
    for i in range(WINDOW, len(data)):
        X.append(data[i - WINDOW : i])
        y.append(data[i])

    X, y   = np.array(X), np.array(y)
    split  = int(len(X) * 0.8)
    model  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X[:split], y[:split])
    joblib.dump(model, f"models/{zone}_rf.pkl")
    print(f"    ✅ models/{zone}_rf.pkl saved")


# ── Train LSTM ────────────────────────────────────────────────────────────────
def train_lstm(series: pd.Series, zone: str):
    print(f"  Training LSTM for {zone}...")
    device = get_device()
    data   = series.values.reshape(-1, 1)

    scaler      = MinMaxScaler()
    data_scaled = scaler.fit_transform(data).flatten()

    X, y = [], []
    for i in range(WINDOW, len(data_scaled)):
        X.append(data_scaled[i - WINDOW : i])
        y.append(data_scaled[i])

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1).to(device)

    dataset    = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model     = LSTMModel(32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        model.train()
        total = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"    Epoch {epoch}/5  loss: {total/len(dataloader):.6f}")

    torch.save(model.state_dict(), f"models/{zone}_lstm.pt")
    joblib.dump(scaler, f"models/{zone}_lstm_scaler.pkl")
    print(f"    ✅ models/{zone}_lstm.pt saved")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}\n")

    for zone in ZONES:
        path = f"data/{zone}.csv"
        if not os.path.exists(path):
            print(f"⚠️  {path} not found — run fetch_training_data.py first\n")
            continue

        print(f"\n{'='*50}")
        print(f"Zone: {zone.upper()}")
        print(f"{'='*50}")

        df     = pd.read_csv(path, index_col="DATE", parse_dates=True)
        series = df[VARIABLE].dropna().resample("D").mean().interpolate()
        print(f"  Data: {len(series)} days ({series.index[0].date()} → {series.index[-1].date()})")

        train_sarima(series, zone)
        train_rf(series, zone)
        train_lstm(series, zone)

    # Save shared config
    joblib.dump(WINDOW, "models/window.pkl")
    print(f"\n✅ All models trained and saved to models/")
