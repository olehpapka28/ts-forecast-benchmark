import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime, timezone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima


def run_model(name, data_path, output_path, horizon, seasonal_periods, value_column, seasonal):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["value"] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna()

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    train = df[:-horizon].copy()
    test = df[-horizon:].copy()

    results = []

    def save_forecast(model_name, preds):
        forecast_df = pd.DataFrame({
            "date": test["date"].values,
            "forecast": preds
        })
        forecast_df.to_csv(RESULTS_DIR / f"forecast_{model_name}_{name}.csv", index=False)

    # ========== Naïve ==========
    last_value = train["value"].iloc[-1]
    test["naive"] = last_value
    y_true = test["value"].values
    y_pred = test["naive"].values
    save_forecast("naive", y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["value"])))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    results.append({
        "model": "naive", "rmse": rmse, "mase": mase, "smape": smape,
        "horizon": horizon, "timestamp": datetime.now(timezone.utc).isoformat()
    })
    print(f"✅ [{name}] Naïve завершено")

    # ========== Seasonal Naïve ==========
    test["seasonal_naive"] = df["value"].shift(seasonal_periods).iloc[-horizon:].values
    y_pred = test["seasonal_naive"].values
    save_forecast("seasonal_naive", y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["value"])))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    results.append({
        "model": "seasonal_naive", "rmse": rmse, "mase": mase, "smape": smape,
        "horizon": horizon, "timestamp": datetime.now(timezone.utc).isoformat()
    })
    print(f"✅ [{name}] Seasonal Naïve завершено")

    # ========== ETS ==========
    ets_model = ExponentialSmoothing(
        train["value"],
        trend="add",
        seasonal="add" if seasonal else None,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated"
    ).fit()

    forecast = ets_model.forecast(steps=horizon)
    y_pred = forecast.values
    save_forecast("ets_add_add", y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["value"])))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    results.append({
        "model": "ets_add_add", "rmse": rmse, "mase": mase, "smape": smape,
        "horizon": horizon, "timestamp": datetime.now(timezone.utc).isoformat()
    })
    print(f"✅ [{name}] ETS завершено")

    # ========== Auto-ARIMA ==========
    arima_model = auto_arima(
        train["value"],
        seasonal=seasonal,
        m=seasonal_periods if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    forecast = arima_model.predict(n_periods=horizon)
    y_pred = forecast
    save_forecast("auto_arima", y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(train["value"])))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    results.append({
        "model": "auto_arima", "rmse": rmse, "mase": mase, "smape": smape,
        "horizon": horizon, "timestamp": datetime.now(timezone.utc).isoformat()
    })
    print(f"✅ [{name}] Auto-ARIMA завершено")

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"📄 [{name}] Результати збережено у {output_path}")


# ========== DATASET RUNS ==========

def run_usd_uah():
    run_model(
        name="usd",
        data_path="data/usd_uah_daily.csv",
        output_path="results/py_naive_usd.csv",
        horizon=365,
        seasonal_periods=1,
        value_column="rate",
        seasonal=False
    )

def run_cpi():
    run_model(
        name="cpi",
        data_path="data/cpi_monthly.csv",
        output_path="results/py_naive_cpi.csv",
        horizon=12,
        seasonal_periods=12,
        value_column="value",
        seasonal=True
    )

def run_brent():
    run_model(
        name="brent",
        data_path="data/brent.csv",
        output_path="results/py_naive_brent.csv",
        horizon=30,
        seasonal_periods=5,  # приблизно тижнева сезонність
        value_column="value",
        seasonal=False  # не всі дні, тому краще без сезонності
    )

if __name__ == "__main__":
    run_usd_uah()
    run_cpi()
    run_brent()
