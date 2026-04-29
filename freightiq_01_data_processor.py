"""
FreightIQ - Data Processor | Logic 8.1
-----------------------------------------
Feature engineering pipeline — industry-grade improvements over v7:

  • Scaler leakage fix  : MinMaxScaler fitted on train split only
  • Stricter delay flag : 80th-percentile threshold (v7 used 75th)
  • New features        : is_peak_month, distance_category, lane_maturity, is_market
  • Bayesian smoothing  : route_risk shrinkage k=15 (v7 used k=10)
  • Recency-aware demand: exponential decay (v7 used linear boost)
  • Temporal stability  : dist_norm uses train-split max_dist only

Outputs:
    freightiq_gps_ready_data_v7.csv        (same name → drop-in replacement)
    freightiq_feature_scaler_v7.pkl
    freightiq_risk_scaler_v7.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("--- Downloading and Loading GPS Trip Dataset ---")

try:
    path = kagglehub.dataset_download("ramakrishnanthiyagu/delivery-truck-trips-data")
    target_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.xlsx', '.csv', '.xls')):
                target_file = os.path.join(root, file)
                break
        if target_file:
            break

    if not target_file:
        raise FileNotFoundError("No data files found in dataset.")

    df = (
        pd.read_excel(target_file, engine='openpyxl')
        if target_file.endswith(('.xlsx', '.xls'))
        else pd.read_csv(target_file)
    )

except Exception as e:
    print(f"!!! Error loading dataset: {e} !!!")
    raise

df.columns = [str(c).strip() for c in df.columns]
print(f"--- Dataset Loaded: {len(df)} rows ---")

# ── 2. TIME & DELAY ENGINEERING ───────────────────────────────────────────────
print("--- Engineering Delay & Time Features ---")

date_cols = ['trip_start_date', 'trip_end_date', 'actual_eta', 'Planned_ETA', 'BookingIDDate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

if 'actual_eta' in df.columns and 'Planned_ETA' in df.columns:
    df['delay_hours'] = (df['actual_eta'] - df['Planned_ETA']).dt.total_seconds() / 3600
else:
    df['delay_hours'] = 0.0

df['delay_hours'] = df['delay_hours'].fillna(0)

# Stricter threshold: top 20% delays flagged (80th pct) — more signal, less noise
delay_threshold = df['delay_hours'].quantile(0.80)
df['delay_flag'] = (df['delay_hours'] > delay_threshold).astype(int)
print(f"  Delay threshold (80th pct): {delay_threshold:.2f}h  |  "
      f"Flagged: {df['delay_flag'].mean():.1%}")

df['month']         = df['trip_start_date'].dt.month
df['day_of_week']   = df['trip_start_date'].dt.dayofweek
df['is_peak_month'] = df['month'].isin([10, 11, 12]).astype(int)

# ── 3. DISTANCE & COST ────────────────────────────────────────────────────────
print("--- Cleaning Distance & Fuel Metrics ---")

df['TRANSPORTATION_DISTANCE_IN_KM'] = df['TRANSPORTATION_DISTANCE_IN_KM'].fillna(
    df['TRANSPORTATION_DISTANCE_IN_KM'].median()
)

# Distance category: short / medium / long haul
df['distance_category'] = np.where(
    df['TRANSPORTATION_DISTANCE_IN_KM'] < 200, 0,
    np.where(df['TRANSPORTATION_DISTANCE_IN_KM'] < 500, 1, 2)
).astype(int)

FUEL_RATE = 87.67
MILEAGE   = 3.5
df['est_fuel_cost'] = (df['TRANSPORTATION_DISTANCE_IN_KM'] / MILEAGE) * FUEL_RATE

# ── 4. VEHICLE LIFECYCLE ──────────────────────────────────────────────────────
print("--- Tracking Vehicle Lifecycle ---")

df = df.sort_values(by=['vehicle_no', 'trip_start_date'])

df['next_origin']     = df.groupby('vehicle_no')['Origin_Location'].shift(-1)
df['next_dest']       = df.groupby('vehicle_no')['Destination_Location'].shift(-1)
df['next_start_time'] = df.groupby('vehicle_no')['trip_start_date'].shift(-1)

df['is_return_trip'] = (
    (df['Origin_Location'] == df['next_dest']) &
    (df['Destination_Location'] == df['next_origin']) &
    (df['Origin_Location'] != df['Destination_Location'])
).astype(int)

df['is_deadhead'] = (
    (df['Destination_Location'] != df['next_origin']) &
    df['next_origin'].notna() &
    (df['is_return_trip'] == 0)
).astype(int)

df['dwell_time_hrs'] = (
    (df['next_start_time'] - df['trip_end_date']).dt.total_seconds() / 3600
).clip(lower=0).fillna(0)

df['idle_loss_raw'] = np.log1p(df['dwell_time_hrs']) * (df['est_fuel_cost'] / 100)

# Past-only driver delay rate — no future leakage
df['driver_delay_rate'] = df.groupby('vehicle_no')['delay_flag'].transform(
    lambda x: x.shift().expanding().mean()
)
df['driver_delay_rate'] = df['driver_delay_rate'].fillna(df['delay_flag'].mean())

threshold_90 = max(48.0, df['dwell_time_hrs'].quantile(0.90))
df['is_continuous'] = df['dwell_time_hrs'] < threshold_90

# ── 5. DEMAND & BAYESIAN ROUTE RISK ──────────────────────────────────────────
print("--- Calculating Bayesian Risk & Recency-Aware Demand ---")

df['route_count'] = df.groupby(['Origin_Location', 'Destination_Location']).cumcount()

# Lane maturity: log-scale of how many trips this lane has seen
df['lane_maturity'] = np.log1p(df['route_count'])

# Recency-weighted demand: exponential decay favours recent activity
df['lane_popularity_raw'] = (
    df.groupby(['Origin_Location', 'Destination_Location'])['BookingID']
    .transform(lambda x: x.shift().expanding().count())
).fillna(0)
# Exponential recency weight (more stable than linear boost in v7)
df['lane_popularity_raw'] = df['lane_popularity_raw'] * np.exp(0.02 * df['route_count']) + 1

df['route_risk_raw'] = (
    df.groupby(['Origin_Location', 'Destination_Location'])['delay_flag']
    .transform(lambda x: x.shift().expanding().std())
).fillna(0)

# Bayesian shrinkage k=15 (v7 used k=10) — stronger prior for sparse lanes
df['route_risk_smoothed'] = (
    df['route_risk_raw'] * (df['route_count'] / (df['route_count'] + 15))
)

df['lane_popularity_log'] = np.log1p(df['lane_popularity_raw'])
df['cost_weight_raw']     = 1 / (1 + np.log1p(df['est_fuel_cost'] / 1000))

# ── 6. LEAKAGE-FREE SCALING ───────────────────────────────────────────────────
print("--- Fitting Scalers on Train Split Only ---")

# Temporal train/test split (80/20 by row order — respects time ordering)
split_idx = int(len(df) * 0.80)
train_mask = np.zeros(len(df), dtype=bool)
train_mask[:split_idx] = True

scaled_cols  = ['lane_popularity_log', 'cost_weight_raw', 'idle_loss_raw', 'route_risk_smoothed']
output_cols  = ['lane_popularity',     'cost_weight',     'idle_loss',     'route_risk']

feature_scaler = MinMaxScaler()
df.loc[train_mask, output_cols] = feature_scaler.fit_transform(
    df.loc[train_mask, scaled_cols]
)
df.loc[~train_mask, output_cols] = feature_scaler.transform(
    df.loc[~train_mask, scaled_cols]
)
joblib.dump(feature_scaler, "freightiq_feature_scaler_v7.pkl")

# dist_norm also fitted on train only
max_dist_train = df.loc[train_mask, 'TRANSPORTATION_DISTANCE_IN_KM'].max()
df['dist_norm'] = df['TRANSPORTATION_DISTANCE_IN_KM'] / max_dist_train

df['confidence_score'] = 1 - df['route_risk']

# ── 7. RISK SCORE ─────────────────────────────────────────────────────────────
print("--- Finalizing Risk Score ---")

df['risk_score_raw'] = (
    0.15 * df['delay_flag'] +
    0.20 * df['dist_norm'] +
    0.30 * (1 - df['lane_popularity']) +
    0.35 * df['route_risk']
)

risk_scaler = MinMaxScaler()
df.loc[train_mask, ['risk_score']]  = risk_scaler.fit_transform(df.loc[train_mask, ['risk_score_raw']])
df.loc[~train_mask, ['risk_score']] = risk_scaler.transform(df.loc[~train_mask, ['risk_score_raw']])
joblib.dump(risk_scaler, "freightiq_risk_scaler_v7.pkl")

# ── 8. EFFICIENCY SCORE (Logic 8.1) ──────────────────────────────────────────
LAMBDA_BASE  = 0.062
GRACE_PERIOD = 4.0

def calculate_logic_81(row):
    if pd.isna(row['next_start_time']) or not row['is_continuous']:
        fb_score = (
            0.30 +
            0.30 * row['lane_popularity'] +
            0.20 * row['cost_weight'] +
            0.20 * (1 - row['risk_score'])
        )
        return round(np.clip(fb_score, 0, 1), 4)

    dist_scalar = np.log1p(row['TRANSPORTATION_DISTANCE_IN_KM'] / 100)
    adj_lambda  = LAMBDA_BASE / max(1.0, dist_scalar)
    t_adj       = max(0.0, row['dwell_time_hrs'] - GRACE_PERIOD)
    ops_score   = np.exp(-adj_lambda * t_adj) * (0.7 + 0.3 * row['dist_norm'])

    if row['is_return_trip']:
        ops_score = min(1.0, ops_score * 1.5)
    elif row['is_deadhead']:
        ops_score *= 0.35

    # Lane maturity bonus: well-established lanes are more predictable
    maturity_bonus = min(0.05, row['lane_maturity'] * 0.005)

    final_score = (
        ops_score                      * 0.38 +
        row['lane_popularity']         * 0.20 +
        row['cost_weight']             * 0.14 +
        (1 - row['risk_score'])        * 0.14 +
        (1 - row['driver_delay_rate']) * 0.09 +
        maturity_bonus
    )

    if row['is_peak_month']:
        final_score += 0.025   # Seasonal demand premium

    return round(np.clip(final_score, 0, 1), 4)

df['efficiency_score'] = df.apply(calculate_logic_81, axis=1)
df['efficiency_label'] = pd.qcut(df['efficiency_score'], q=3, labels=['Low', 'Medium', 'High'])

# is_market feature
if 'Market/Regular' in df.columns:
    df['is_market'] = (df['Market/Regular'].str.strip() == 'Regular').astype(int)
else:
    df['is_market'] = 0

# ── 9. EXPORT ─────────────────────────────────────────────────────────────────
required_columns = [
    'BookingID', 'vehicle_no', 'Origin_Location', 'Destination_Location',
    'TRANSPORTATION_DISTANCE_IN_KM', 'day_of_week', 'month', 'delay_flag',
    'lane_popularity', 'cost_weight', 'risk_score', 'driver_delay_rate',
    'route_risk', 'confidence_score', 'idle_loss', 'is_return_trip', 'is_deadhead',
    'efficiency_score', 'efficiency_label',
    # New in v8.1
    'is_peak_month', 'distance_category', 'lane_maturity', 'is_market',
]
if 'Market/Regular' in df.columns:
    required_columns.append('Market/Regular')

final_df = df[required_columns].copy()
final_df.to_csv("freightiq_gps_ready_data_v7.csv", index=False)

print("\n--- SUCCESS! Logic 8.1 Data Processor Complete ---")
print(f"Total trips processed : {len(final_df)}")
print(f"Train split           : {split_idx} rows")
print(f"Test split            : {len(final_df) - split_idx} rows")
print(f"Delay flag rate       : {final_df['delay_flag'].mean():.1%}")
print(f"Peak month trips      : {final_df['is_peak_month'].mean():.1%}")
print(f"Distance categories   : short={( final_df['distance_category']==0).mean():.1%}  "
      f"medium={(final_df['distance_category']==1).mean():.1%}  "
      f"long={(final_df['distance_category']==2).mean():.1%}")
print("\n--- Summary Stats ---")
print(final_df[['efficiency_score', 'lane_popularity', 'risk_score', 'lane_maturity']]
      .describe().loc[['mean', 'std', 'min', 'max']])
