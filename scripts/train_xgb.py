import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Utils ---
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_metrics(y_true, y_pred, model_name="XGBoost"):
    print(f"{model_name} MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"{model_name} MAPE: {mape(y_true, y_pred)*100:.2f}%")
    print(f"{model_name} R2  : {r2_score(y_true, y_pred):.2f}")

# --- Config ---
DATA_PATH = "data/world_bank_data_2025.csv"
TARGET = "GDP per Capita (Current USD)"

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df['year'] = df['year'].astype(int)

# --- Feature engineering: lags & changes ---
def add_lags_and_deltas(df, cols, group='country_id', n_lag=1):
    out = df.copy()
    g = out.groupby(group, group_keys=False)
    for c in cols:
        out[f"{c}_lag{n_lag}"] = g[c].shift(n_lag)
        out[f"{c}_chg{n_lag}"] = out[c] - out[f"{c}_lag{n_lag}"]
    return out

feat_cols = [c for c in df.columns if c not in ['country_name','country_id','year']]
df_lag = add_lags_and_deltas(df, feat_cols, n_lag=1)
df_lag = df_lag.dropna()

X = df_lag[[c for c in df_lag.columns if c.endswith('_lag1') or c.endswith('_chg1')]]
y = df_lag[TARGET]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Train model ---
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)
xgb.fit(X_train, y_train)

# --- Evaluate ---
y_pred = xgb.predict(X_test)
print_metrics(y_test, y_pred, "XGBoost")