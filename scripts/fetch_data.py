#!/usr/bin/env python3
"""
Завантажує:
1) Щоденний курс USD/UAH з API НБУ
2) Щомісячний CPI з SDMX-API Держстату
Зберігає у data/ два CSV і друкує їхні SHA-256.
"""
import hashlib, pathlib, datetime as dt, requests, csv, io, zipfile, textwrap
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------- 1. USD/UAH (з перезапуском) --------------------------
print("Завантаження архіву курсу USD/UAH з НБУ...")
import time

start_date = dt.date(2010, 1, 1)
end_date = dt.date.today()
delta = dt.timedelta(days=1)

csv_usd = DATA_DIR / "usd_uah_daily.csv"
existing_dates = set()

# Якщо файл існує — зчитуємо вже завантажені дні
if csv_usd.exists():
    with csv_usd.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            existing_dates.add(row[0])

print(f"Пропускаємо вже наявні {len(existing_dates)} днів.")

with csv_usd.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if not existing_dates:
        w.writerow(["date", "rate"])  # перший рядок

    current_date = start_date
    while current_date <= end_date:
        date_str_fmt = current_date.strftime("%d.%m.%Y")
        if date_str_fmt in existing_dates:
            current_date += delta
            continue
        url = f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode=USD&date={current_date.strftime('%Y%m%d')}&json"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                w.writerow([data[0]["exchangedate"], data[0]["rate"]])
                print(f"{data[0]['exchangedate']} → {data[0]['rate']}")
            else:
                print(f"{current_date} → пусто")
        except Exception as e:
            print(f"⚠️ {current_date}: помилка — {e}")
        current_date += delta
        time.sleep(0.2)  # затримка між запитами

# ---------- 2. CPI (з SDMX Держстат) ---------------------------------------
print("Запит → CPI з Держстат SDMX …")
from pandasdmx import Request

estat = Request('estat')
resp = estat.dataflow()
flow = resp.dataflow['prc_hicp_midx']   # індекс споживчих цін

# [!] Тут складна фільтрація — краще поки перейти до ручного джерела або csv
raise NotImplementedError("SDMX-інтеграцію реалізуємо трохи згодом.")

# ---------- 3. SHA-256 ------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]

print("\nКонтрольні суми:")
print(f"  {csv_usd.name}: {sha256(csv_usd)}")
print(f"  {csv_cpi.name}: {sha256(csv_cpi)}")
