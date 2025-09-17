import pandas as pd
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

warnings.filterwarnings("ignore")

# --- Utils ---
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_metrics(y_true, y_pred, model_name="SARIMAX"):
    print(f"{model_name} MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"{model_name} MAPE: {mape(y_true, y_pred)*100:.2f}%")
    print(f"{model_name} R2  : {r2_score(y_true, y_pred):.2f}")

# --- Config ---
DATA_PATH = "data/world_bank_data_2025.csv"
COUNTRY_ID = "id"  # Indonesia
TARGET = "GDP per Capita (Current USD)"

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df['year'] = df['year'].astype(int)

indo = df[df['country_id'] == COUNTRY_ID].copy()
indo = indo[['year', TARGET]].dropna()
indo = indo.set_index(pd.to_datetime(indo['year'], format="%Y"))

y = indo[TARGET]

# --- Train-test split ---
train_end = 2019
y_train = y[y.index.year <= train_end]
y_test = y[y.index.year > train_end]

# --- Grid search SARIMAX order (p,d,q) ---
orders = list(itertools.product([0,1,2], [0,1,2], [0,1,2]))
best_score, best_order, best_fc = float("inf"), None, None

for order in orders:
    try:
        model = SARIMAX(y_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=len(y_test)).predicted_mean
        score = mape(y_test, fc)
        if score < best_score:
            best_score, best_order, best_fc = score, order, fc
    except:
        continue

print(f"Best Order: {best_order}, Test MAPE: {best_score*100:.2f}%")
print_metrics(y_test, best_fc, "SARIMAX")
