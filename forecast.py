"""
forecast.py
-----------
Loads the correct pretrained models for a climate zone,
runs backtesting, and generates a 7-day forecast.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WINDOW = 30


# ── LSTM architecture (must match train_all.py) ───────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def _metrics(actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return mae, rmse


def _future_dates(last_date, n=7):
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n, freq="D")


# ── SARIMA forecast ───────────────────────────────────────────────────────────
def forecast_sarima(series: pd.Series, zone: str, forecast_days: int = 7):
    model_fit = joblib.load(f"models/{zone}_sarima.pkl")

    split         = int(len(series) * 0.8)
    test          = series.iloc[split:]
    backtest_pred = model_fit.forecast(steps=len(test))
    mae, rmse     = _metrics(test.values, backtest_pred.values)

    # Refit on full series for live forecast
    full_fit = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    future_pred  = full_fit.forecast(steps=forecast_days).values
    future_dates = _future_dates(series.index[-1], forecast_days)

    return test.values, backtest_pred.values, mae, rmse, future_dates, future_pred


# ── RF forecast ───────────────────────────────────────────────────────────────
def forecast_rf(series: pd.Series, zone: str, forecast_days: int = 7):
    model = joblib.load(f"models/{zone}_rf.pkl")
    data  = series.values

    X, y = [], []
    for i in range(WINDOW, len(data)):
        X.append(data[i - WINDOW : i])
        y.append(data[i])
    X, y  = np.array(X), np.array(y)

    split         = int(len(X) * 0.8)
    backtest_pred = model.predict(X[split:])
    mae, rmse     = _metrics(y[split:], backtest_pred)

    # Rolling future forecast
    last_window = data[-WINDOW:].tolist()
    future_pred = []
    for _ in range(forecast_days):
        x_in     = np.array(last_window[-WINDOW:]).reshape(1, -1)
        next_val = model.predict(x_in)[0]
        future_pred.append(next_val)
        last_window.append(next_val)

    future_dates = _future_dates(series.index[-1], forecast_days)
    return y[split:], backtest_pred, mae, rmse, future_dates, np.array(future_pred)


# ── LSTM forecast ─────────────────────────────────────────────────────────────
def forecast_lstm(series: pd.Series, zone: str, forecast_days: int = 7):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    scaler = joblib.load(f"models/{zone}_lstm_scaler.pkl")

    lstm_model = LSTMModel(32).to(device)
    lstm_model.load_state_dict(
        torch.load(f"models/{zone}_lstm.pt", map_location=device)
    )
    lstm_model.eval()

    data        = series.values.reshape(-1, 1)
    data_scaled = scaler.transform(data).flatten()

    X, y = [], []
    for i in range(WINDOW, len(data_scaled)):
        X.append(data_scaled[i - WINDOW : i])
        y.append(data_scaled[i])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
    y = np.array(y)

    split  = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]

    with torch.no_grad():
        pred_scaled = lstm_model(X_test).cpu().numpy().flatten()

    backtest_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual        = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae, rmse     = _metrics(actual, backtest_pred)

    # Rolling future forecast
    last_window = data_scaled[-WINDOW:].tolist()
    future_pred = []
    with torch.no_grad():
        for _ in range(forecast_days):
            x_in     = torch.tensor(
                np.array(last_window[-WINDOW:]), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1).to(device)
            next_val = lstm_model(x_in).cpu().numpy().flatten()[0]
            future_pred.append(next_val)
            last_window.append(next_val)

    future_pred  = scaler.inverse_transform(
        np.array(future_pred).reshape(-1, 1)
    ).flatten()
    future_dates = _future_dates(series.index[-1], forecast_days)

    return actual, backtest_pred, mae, rmse, future_dates, future_pred


# ── Main entry point ──────────────────────────────────────────────────────────
def run_all_models(live_df: pd.DataFrame, zone: str, variable: str = "TG"):
    """
    Run all three models for a given zone and live DataFrame.

    Returns dict of results keyed by model name.
    """
    series = live_df[variable].dropna().resample("D").mean().interpolate()

    results = {}
    for name, fn in [("SARIMA", forecast_sarima),
                     ("RF",     forecast_rf),
                     ("LSTM",   forecast_lstm)]:
        try:
            actual, backtest, mae, rmse, fdates, fpred = fn(series, zone)
            results[name] = {
                "actual":       actual,
                "backtest":     backtest,
                "mae":          mae,
                "rmse":         rmse,
                "future_dates": fdates,
                "future_pred":  fpred,
            }
        except Exception as e:
            print(f"⚠️  {name} failed for {zone}: {e}")

    return results
