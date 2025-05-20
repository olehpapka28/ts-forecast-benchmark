import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

def plot_forecast(data_path, result_path, title, value_col="value", date_col="date"):
    df = pd.read_csv(data_path, parse_dates=[date_col])
    results = pd.read_csv(result_path)

    df = df.sort_values(date_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna()

    horizon = int(results["horizon"].iloc[0])
    test = df[-horizon:].copy()

    # Підготовка прогнозів
    forecast_df = pd.DataFrame({date_col: test[date_col]})
    for model in results["model"]:
        forecast_file = Path(f"results/forecast_{model}_{result_path.stem.split('_')[-1]}.csv")
        if forecast_file.exists():
            fcast = pd.read_csv(forecast_file)
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

    # Збереження
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r'[^a-zA-Z0-9_]+', '_', title.lower())
    file_name = f"{safe_title}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=150)
    print(f"📉 Графік збережено → {out_path}")


# Побудова для Біткоіна 
plot_forecast(
    data_path=Path("data/usd_uah_daily.csv"),
    result_path=Path("results/py_naive_usd.csv"),
    title="Прогноз курсу USD/UAH",
    value_col="rate"
)

# Побудова для CPI
plot_forecast(
    data_path=Path("data/cpi_monthly.csv"),
    result_path=Path("results/py_naive_cpi.csv"),
    title="Прогноз ІСЦ (CPI)",
    value_col="value"
)

# Побудова для Brent
plot_forecast(
    data_path=Path("data/brent.csv"),
    result_path=Path("results/py_naive_brent.csv"),
    title="Прогноз ціни Brent crude oil",
    value_col="value"
)
