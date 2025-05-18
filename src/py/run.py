import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime, timezone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import os

# ========== Конфігурація ==========
DATA_PATH = Path("data/usd_uah_daily.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "py_naive_usd.csv"
TEST_HORIZON = 365
SEASONALITY = 365

# ========== Завантаження та підготовка ==========
df = pd.read_csv(DATA_PATH, parse_dates=["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df = df.dropna()

train = df[:-TEST_HORIZON].copy()
test = df[-TEST_HORIZON:].copy()

results = []

# ========== Naïve ==========
last_value = train["rate"].iloc[-1]
test["naive"] = last_value

y_true = test["rate"].values
y_pred = test["naive"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["rate"])))
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

results.append({
    "model": "naive",
    "rmse": round(rmse, 4),
    "mase": round(mase, 4),
    "smape": round(smape, 2),
    "horizon": TEST_HORIZON,
    "timestamp": datetime.now(timezone.utc).isoformat()
})

print("✅ Naïve модель завершено")

# ========== Seasonal Naïve ==========
test["seasonal_naive"] = df["rate"].shift(SEASONALITY).iloc[-TEST_HORIZON:].values
y_pred = test["seasonal_naive"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["rate"])))
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

results.append({
    "model": "seasonal_naive",
    "rmse": round(rmse, 4),
    "mase": round(mase, 4),
    "smape": round(smape, 2),
    "horizon": TEST_HORIZON,
    "timestamp": datetime.now(timezone.utc).isoformat()
})

print("✅ Seasonal Naïve модель завершено")

# ========== ETS (Exponential Smoothing) ==========
ets_model = ExponentialSmoothing(
    train["rate"],
    trend="add",
    seasonal="add",
    seasonal_periods=SEASONALITY,
    initialization_method="estimated"
).fit()

forecast = ets_model.forecast(steps=TEST_HORIZON)
test["ets"] = forecast.values

y_pred = forecast.values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["rate"])))
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

results.append({
    "model": "ets_add_add",
    "rmse": round(rmse, 4),
    "mase": round(mase, 4),
    "smape": round(smape, 2),
    "horizon": TEST_HORIZON,
    "timestamp": datetime.now(timezone.utc).isoformat()
})

print("✅ ETS модель завершено")

# ========== Auto-ARIMA ==========
arima_model = auto_arima(
    train["rate"],
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

forecast = arima_model.predict(n_periods=TEST_HORIZON)
test["arima"] = forecast

y_pred = forecast

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["rate"])))
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

results.append({
    "model": "auto_arima",
    "rmse": round(rmse, 4),
    "mase": round(mase, 4),
    "smape": round(smape, 2),
    "horizon": TEST_HORIZON,
    "timestamp": datetime.now(timezone.utc).isoformat()
})

print("✅ Auto-ARIMA модель завершено")

# ==========  ЗАПИС  ==========
out = pd.DataFrame(results)
out.to_csv(OUTPUT_PATH, index=False)
print(f"📄 Результати збережено у {OUTPUT_PATH}")
