import yaml, time, psutil, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from dieboldmariano import dm_test as diebold_mariano
# ...читання YAML, вибір моделі, час/пам'ять профілювання, запис parquet...
print("Hello from Python container!  OK.")
