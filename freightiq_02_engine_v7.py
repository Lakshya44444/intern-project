"""
FreightIQ - XGBoost Scoring Engine | v7
-----------------------------------------
Trains an XGBoost regressor to learn and generalise the rule-based
efficiency_score produced by Logic 7.2.5, enabling real-time prediction
without rerunning the full feature-engineering pipeline.

⚠  delay_flag is a historical label derived from actual vs planned ETA.
   It is valid for training but NOT available for future (unfinished) trips.
   Production inference via app.py must substitute driver_delay_rate as proxy.

Inputs:  freightiq_gps_ready_data_v7.csv
Outputs: freightiq_ai_engine_v7.pkl
         v7_feature_list.pkl
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("--- Loading Logic 7.2.5 Intelligence Dataset ---")
try:
    df = pd.read_csv("freightiq_gps_ready_data_v7.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "Dataset not found. Run freightiq_01_data_processor.py first."
    )

df = df.dropna(subset=['efficiency_score', 'TRANSPORTATION_DISTANCE_IN_KM'])

# ── 2. FEATURE ENCODING ───────────────────────────────────────────────────────
if 'Market/Regular' in df.columns:
    df['is_regular'] = (df['Market/Regular'].str.strip() == 'Regular').astype(int)
else:
    df['is_regular'] = 1

# ── 3. FEATURE SELECTION ──────────────────────────────────────────────────────
FEATURES = [
    'TRANSPORTATION_DISTANCE_IN_KM',
    'day_of_week',
    'month',
    'delay_flag',           # Historical label — see module docstring
    'lane_popularity',
    'risk_score',
    'driver_delay_rate',
    'route_risk',
    'is_return_trip',
    'is_deadhead',
    'is_regular',
]

X = df[FEATURES]
y = df['efficiency_score']

# ── 4. TRAIN-TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. HYPERPARAMETER SEARCH ──────────────────────────────────────────────────
print("--- Tuning XGBoost Regressor ---")

param_grid = {
    'n_estimators':    [500, 1000, 1500],
    'max_depth':       [5, 7, 9],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.8, 0.9],
    'colsample_bytree':[0.8, 0.9],
    'gamma':           [0.1, 0.2],
    'reg_alpha':       [0.1, 0.5],   # L1 regularisation
    'reg_lambda':      [1.5, 2.0],   # L2 regularisation
}

search = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=42),  # n_jobs handled by search
    param_distributions=param_grid,
    n_iter=15,
    scoring='r2',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,   # parallelise across CV folds, not within each estimator
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

# ── 6. EVALUATION ─────────────────────────────────────────────────────────────
print("\n--- Model Evaluation ---")
preds = best_model.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)

print(f"RMSE (Error Margin) : {rmse:.4f}")
print(f"R² Score            : {r2:.4f}")
print(f"Best Params         : {search.best_params_}")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
importance = (
    pd.DataFrame({'feature': FEATURES, 'importance': best_model.feature_importances_})
    .sort_values('importance', ascending=False)
)
print("\nTop 5 Drivers of Logistics Efficiency:")
print(importance.head(5).to_string(index=False))

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
print("\n--- Saving FreightIQ v7 Engine ---")
joblib.dump(best_model, "freightiq_ai_engine_v7.pkl")
joblib.dump(FEATURES,   "v7_feature_list.pkl")

print("--- SUCCESS! freightiq_ai_engine_v7.pkl ready. ---")
