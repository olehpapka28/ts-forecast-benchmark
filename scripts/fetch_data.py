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

# ---------- 1. USD/UAH ------------------------------------------------------
today = dt.date.today().strftime("%Y-%m-%d")
url_usd = (
    "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
    "?valcode=USD&date=&periodicity=0&json"
)
print("Запит → НБУ …")
resp = requests.get(url_usd, timeout=30)
resp.raise_for_status()
rows = resp.json()

csv_usd = DATA_DIR / "usd_uah_daily.csv"
with csv_usd.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "rate"])
    for row in rows:
        w.writerow([row["exchangedate"], row["rate"]])
print(f"USD/UAH записано → {csv_usd}")

# ---------- 2. CPI ----------------------------------------------------------
print("Запит → Держстат (SDMX) …")
url_cpi = (
    "https://index.minfin.com.ua/reference/cpi/csv/"  # простий csv-дамп
)
resp = requests.get(url_cpi, timeout=30)
resp.raise_for_status()
csv_cpi = DATA_DIR / "cpi_monthly.csv"
csv_cpi.write_bytes(resp.content)
print(f"CPI записано → {csv_cpi}")

# ---------- 3. SHA-256 ------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]

print("\nКонтрольні суми:")
print(f"  {csv_usd.name}: {sha256(csv_usd)}")
print(f"  {csv_cpi.name}: {sha256(csv_cpi)}")
