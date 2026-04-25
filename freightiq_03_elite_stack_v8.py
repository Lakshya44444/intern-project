"""
FreightIQ - Elite AI Stack | v8
---------------------------------
Two-stage probabilistic pricing model:

  Stage 1 (Risk Engine)   : XGBoost Classifier  → P(delay | trip features)
  Stage 2 (Pricing Engine): XGBoost Regressor   → Probabilistic Efficiency Score

BUG FIXED — Data Leakage in Naive Stacking:
  The original code called risk_engine.predict_proba(X_cls) on the entire dataset
  after training on X_train_c.  This means the Stage 2 training rows received
  in-sample (overfitted) Stage 1 predictions, inflating Stage 2 R² artificially.

  Fix: use cross_val_predict (out-of-fold / OOF) to generate predicted_delay_prob
  so that every row's probability is predicted by a model that never saw that row.
  The final Risk Engine is then retrained on the full train split for deployment.

Inference pipeline (app.py):
  raw_features → risk_engine.predict_proba → predicted_delay_prob
              → concat → pricing_engine.predict → probabilistic_efficiency

Inputs:  freightiq_gps_ready_data_v7.csv
Outputs: freightiq_risk_engine_v8.pkl
         freightiq_pricing_engine_v8.pkl
         v8_feature_list.pkl
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("--- Loading Intelligence Dataset ---")
try:
    df = pd.read_csv("freightiq_gps_ready_data_v7.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "Dataset not found. Run freightiq_01_data_processor.py first."
    )

df = df.dropna(subset=['TRANSPORTATION_DISTANCE_IN_KM', 'efficiency_score'])

# Base features shared by both stages (delay_flag excluded: it's the Stage 1 target)
BASE_FEATURES = [
    'TRANSPORTATION_DISTANCE_IN_KM',
    'day_of_week',
    'month',
    'lane_popularity',
    'route_risk',
    'driver_delay_rate',
    'is_return_trip',
    'is_deadhead',
]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: RISK ENGINE (Delay Classifier)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STAGE 1: Training Delay Risk Engine ---")

X_cls = df[BASE_FEATURES]
y_cls = df['delay_flag']

RISK_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
)

# OOF predictions: each row is predicted by a fold that excluded it from training.
# This is the correct way to build a stacked feature — prevents leakage into Stage 2.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_delay_probs = cross_val_predict(
    xgb.XGBClassifier(**RISK_PARAMS),
    X_cls, y_cls,
    cv=cv,
    method='predict_proba',
)[:, 1]

df['predicted_delay_prob'] = oof_delay_probs

# Holdout evaluation with a fixed train/test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

risk_engine = xgb.XGBClassifier(**RISK_PARAMS)
risk_engine.fit(X_train_c, y_train_c)

acc = accuracy_score(y_test_c, risk_engine.predict(X_test_c))
auc = roc_auc_score(y_test_c, risk_engine.predict_proba(X_test_c)[:, 1])
print(f"Risk Engine Accuracy : {acc:.4f}")
print(f"Risk Engine AUC Score: {auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: PRICING ENGINE (Probabilistic Efficiency Regressor)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STAGE 2: Building Probabilistic Efficiency Target (Logic 8.0) ---")

# Logic 8.0 vs Logic 7.2.5:
#   v7 slot → cost_weight (static fuel ratio, no predictive power)
#   v8 slot → (1 - predicted_delay_prob) — forward-looking ML signal
#
# Weight allocation:
#   40% ops efficiency  (historical formula base)
#   20% lane demand     (route popularity)
#   15% route risk      (historical variance)
#   15% delay freedom   (ML-predicted — the forward-looking differentiator)
#   10% driver reliability
def calculate_elite_score(row: pd.Series) -> float:
    return round(np.clip(
        0.40 * row['efficiency_score'] +
        0.20 * row['lane_popularity'] +
        0.15 * (1 - row['risk_score']) +
        0.15 * (1 - row['predicted_delay_prob']) +
        0.10 * (1 - row['driver_delay_rate']),
        0, 1,
    ), 4)

df['probabilistic_efficiency'] = df.apply(calculate_elite_score, axis=1)

print("\n--- STAGE 2: Training Pricing Engine ---")

V8_FEATURES = BASE_FEATURES + ['predicted_delay_prob']
X_reg = df[V8_FEATURES]
y_reg = df['probabilistic_efficiency']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

pricing_engine = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
)
pricing_engine.fit(X_train_r, y_train_r)

r2 = r2_score(y_test_r, pricing_engine.predict(X_test_r))
print(f"Pricing Engine R² Score: {r2:.4f}")
print(
    "Note: R² reflects how well the engine approximates the Logic 8.0 formula.\n"
    "      Validate against real pricing outcomes for true generalisation signal."
)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
importance = (
    pd.DataFrame({'feature': V8_FEATURES, 'importance': pricing_engine.feature_importances_})
    .sort_values('importance', ascending=False)
)
print("\nPricing Drivers (importance ranking):")
print(importance.to_string(index=False))

# ── SAVE THE ELITE STACK ──────────────────────────────────────────────────────
print("\n--- Saving Elite AI Stack ---")
joblib.dump(risk_engine,    "freightiq_risk_engine_v8.pkl")
joblib.dump(pricing_engine, "freightiq_pricing_engine_v8.pkl")
joblib.dump(V8_FEATURES,    "v8_feature_list.pkl")

print("\n--- SUCCESS! Elite Stack v8 saved. ---")
print(
    "Inference pipeline:\n"
    "  raw_features\n"
    "  → risk_engine.predict_proba(BASE_FEATURES)[:, 1]  → predicted_delay_prob\n"
    "  → pricing_engine.predict(BASE_FEATURES + [predicted_delay_prob])\n"
    "  → probabilistic_efficiency_score"
)
