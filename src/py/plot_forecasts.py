import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_forecast(data_path, result_path, title, date_col="date", value_col="value"):
    df = pd.read_csv(data_path, parse_dates=[date_col])
    results = pd.read_csv(result_path)

    df = df.sort_values(date_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna()

    horizon = int(results["horizon"].iloc[0])
    test = df[-horizon:].copy()

    # Підготуємо прогнози
    forecast_df = pd.DataFrame({date_col: test[date_col]})
    for model in results["model"]:
        # Очікуємо файл з назвою виду: forecast_{модель}_{ряд}.csv
        path = Path(f"results/forecast_{model}_{result_path.stem.split('_')[-1]}.csv")
        if path.exists():
            fcast = pd.read_csv(path)
            forecast_df[model] = fcast["forecast"].values[:horizon]

    # Побудова графіка
    plt.figure(figsize=(12, 6))
    plt.plot(test[date_col], test[value_col], label="Фактичне", linewidth=2)

    for col in forecast_df.columns:
        if col != date_col:
            plt.plot(forecast_df[date_col], forecast_df[col], label=col)

    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel("Значення")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Запуск прикладу (USD/UAH)
plot_forecast(
    data_path=Path("data/usd_uah_daily.csv"),
    result_path=Path("results/py_naive_usd.csv"),
    title="Прогноз курсу USD/UAH"
)

# Запуск прикладу (CPI)
plot_forecast(
    data_path=Path("data/cpi_monthly.csv"),
    result_path=Path("results/py_naive_cpi.csv"),
    title="Прогноз ІСЦ (CPI)"
)
