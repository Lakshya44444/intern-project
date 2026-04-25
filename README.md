# FreightIQ Elite v8.6 — AI-Driven Logistics Pricing Engine

A two-stage XGBoost system that converts 5 user inputs into a probabilistic freight **price corridor** [P_min, P_max] using physics-based mileage modelling, a time-decay return-load engine, Claude Vision photo extraction, and a 5-layer financial engine.

---

## Table of Contents

1. [Core Vision](#core-vision)
2. [System Architecture](#system-architecture)
3. [File Structure](#file-structure)
4. [How to Run](#how-to-run)
5. [Intelligence Abstraction Layer](#intelligence-abstraction-layer)
6. [The AI Stack](#the-ai-stack)
7. [5-Layer Pricing Engine](#5-layer-pricing-engine)
8. [Model Results](#model-results)
9. [API Reference](#api-reference)
10. [ML Training Scripts](#ml-training-scripts)
11. [Limitations & Caveats](#limitations--caveats)
12. [Dependencies](#dependencies)

---

## Core Vision

Traditional logistics pricing uses **Fixed Rate per KM**. FreightIQ replaces this with a **Pricing Corridor** — a [P_min, P_max] range that changes based on:

- A **time-decay model** for return (backhaul) load probability
- Physics-adjusted **fuel cost** based on actual load weight
- Real-time **demand–supply imbalance** (lane popularity vs. route risk)
- **Competitor benchmarking** that snaps the corridor to ±10%/+12% of market
- A **feasibility gate** ensuring the driver's survival earnings are always covered

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│               STAGE A: VISION & AUTHENTICATION LAYER                │
│                                                                     │
│  POST /api/extract-vision — upload a photo to auto-fill fields      │
│    Diesel board photo  → authentic_diesel (Rs/L)                    │
│    Competitor quote    → authentic_competitor (Rs)                  │
│  Powered by Claude Vision (claude-sonnet-4-6)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ auto-fills form fields
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    USER INPUT (5 fields)                             │
│   Source │ Destination │ Trip Date │ Load Weight │ Vehicle Type     │
│          + Optional: Competitor Price, Diesel Price (from scan)     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STAGE B: INTELLIGENCE ABSTRACTION LAYER               │
│                                                                     │
│  Distance   → Haversine x 1.32 road factor (81 Indian cities)      │
│  Temporal   → Day-of-week + Season from Trip Date                  │
│  Physics    → M_loaded = 3.5 - (0.02 x weight_tonnes)              │
│  Lane KB    → lane_popularity, route_risk from 6,880-trip history   │
│  Arrival    → Departure 06:00 + travel time (50 km/h avg)          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│          TWO-STAGE AI STACK + PROBABILISTIC DECAY ENGINE            │
│                                                                     │
│  Stage 1 — Risk Engine (XGBoost Classifier)  [AUC = 0.88]         │
│  Stage 2 — Pricing Engine (XGBoost Regressor) [R² = 0.91]         │
│                                                                     │
│  p_return — Time-Decay Model                                       │
│    p = e^(-lambda x max(0, T_active - T_grace))                    │
│    FTL:     lambda=0.062  (long-haul stability)                    │
│    Carting: lambda=0.15   (short-haul aggressive decay)            │
│    T_grace=4h, Business window 08:00-20:00                         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STAGE C: 5-LAYER CORRIDOR STACK                       │
│                                                                     │
│  L1 Operational  → C_base = Fuel(A+B) + Tolls(round-trip) + Wages │
│  L2 Risk         → C_risk = (1-p_return) x Fuel_Empty_Return       │
│  L3 Corridor     → P_min = C_base+C_risk / P_max = P_min+premia   │
│  L4 Market Snap  → Ceiling: min(P_max, Competitor x 1.12)          │
│                 → Floor:   max(P_min, Competitor x 0.90)           │
│  L5 Gate         → Recommended = max(Floor, C_survival)            │
│                    Status: Healthy / Low Margin / Infeasible        │
│                                                                     │
│  OUTPUT → Corridor [P_min, P_max] + Recommended + Status           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
freightiq/
│
├── app.py                               # Flask REST API — Intelligence + 5-Layer pricing
├── requirements.txt                     # Python dependencies
│
├── freightiq_01_data_processor.py       # Logic 7.2.5 — GPS feature engineering
├── freightiq_02_engine_v7.py            # XGBoost v7 — historical efficiency scorer
├── freightiq_03_elite_stack_v8.py       # Elite Stack v8 — two-stage probabilistic model
│
├── freightiq_gps_ready_data_v7.csv      # Engineered dataset (6,880 trips) — Lane KB source
├── freightiq_risk_engine_v8.pkl         # Stage 1: XGBoost delay classifier
├── freightiq_pricing_engine_v8.pkl      # Stage 2: XGBoost efficiency regressor
├── freightiq_ai_engine_v7.pkl           # v7 historical scorer (reference)
├── freightiq_feature_scaler_v7.pkl      # MinMaxScaler for lane/cost/idle/risk features
├── freightiq_risk_scaler_v7.pkl         # MinMaxScaler for composite risk score
├── v7_feature_list.pkl                  # Feature names for v7 model
├── v8_feature_list.pkl                  # Feature names for v8 models
│
└── frontend/                            # React + Vite + Tailwind UI
    ├── package.json
    ├── vite.config.js                   # Dev server: port 3000, proxy → Flask :5000
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx                      # Root — layout, state, fetch handler
        ├── index.css                    # Tailwind + glassmorphism utilities
        └── components/
            ├── Header.jsx               # Logo, v8.6 badge, AUC/R² stat pills
            ├── InputForm.jsx            # 4-field form: cities, date, weight, competitor
            └── ResultDashboard.jsx      # Abstraction panel, confidence gauge, 5-layer breakdown
```

---

## How to Run

> **Both servers must run simultaneously.** Open two terminals.

### Install Python dependencies (first time only)
```bash
pip install -r requirements.txt
```

### Terminal 1 — Flask API (port 5000)
```bash
cd e:/final_intern
python app.py
```
Expected output: `FreightIQ Elite v8.6 — API running at http://localhost:5000`

Verify: open `http://localhost:5000/api/health` → `{"status": "ok", "version": "v8.6"}`

### Terminal 2 — React UI (port 3000)
```bash
cd e:/final_intern/frontend
npm install        # first time only
npm run dev
```
Expected output: `Local: http://localhost:3000`

### Open in browser
```
http://localhost:3000
```

> **Note:** `localhost:5000` shows "Not Found" in the browser — that is correct. The Flask API has no HTML homepage, only JSON endpoints. Always use `localhost:3000` for the dashboard.

---

## Intelligence Abstraction Layer

The user enters 4 fields. The system auto-derives everything else before ML inference.

### Distance Engine

```
distance = haversine(origin_coords, dest_coords) × 1.32  (road factor)
```

81 Indian cities are pre-loaded with GPS coordinates. If both cities are recognised, Haversine × 1.32 is used. If not, the historical average distance for that lane is used from the CSV. Falls back to 500 km if neither is available.

### Temporal Features

Extracted from the `trip_date` (YYYY-MM-DD):

| Month | Season | Urgency Premium (L3) |
|---|---|---|
| Oct, Nov, Dec | Peak | +18% × C_base |
| Jan, Feb | Winter | +5% × C_base |
| Mar | Pre-Summer | 0% |
| Apr, May, Jun | Summer | 0% (no premium on P_max) |
| Jul, Aug, Sep | Monsoon | 0% (no premium on P_max) |

### Physics Engine

```
M_loaded = 3.5 − (0.02 × weight_tonnes)   [min 1.5 km/L]
M_empty  = 4.8 km/L  (fixed, Uttarakhand conditions)
Diesel   = Rs 87.67/L  (default; overridable via vision photo)
```

| Weight | M_loaded |
|---|---|
| 5T | 3.40 km/L |
| 10T | 3.30 km/L |
| 20T | 3.10 km/L |
| 30T | 2.90 km/L |

### Lane Knowledge Base

Derived from the 6,880-trip GPS dataset. For each origin–destination pair:

| Signal | Source | Use |
|---|---|---|
| `lane_popularity` | Recency-weighted booking count | Demand premium + p_return decay seed |
| `route_risk` | Bayesian-smoothed delay variance | Supply signal in L3 |
| `driver_delay_rate` | Expanding past delay mean | ML input feature |

### Arrival Time & Time-Decay Clock

Assuming 06:00 departure:

```
arrival_time  = departure(06:00) + distance / 50 km/h
expected_dwell = 4h (grace) + (1 - lane_popularity) × 20h
T_active      = business-hours adjusted dwell (08:00-20:00 window)
```

High-popularity lanes (0.8+) expect ~8h dwell → short T_active → high p_return.
Low-popularity lanes (0.1) expect ~28h dwell → long T_active → low p_return.

---

## The AI Stack

### Stage 1 — Risk Engine (XGBoost Classifier)

Predicts `p_delay` — probability that the trip will be delayed.

**Input features (8):**
```
TRANSPORTATION_DISTANCE_IN_KM, day_of_week, month,
lane_popularity, route_risk, driver_delay_rate,
is_return_trip, is_deadhead
```

**Results:**

| Metric | Value |
|---|---|
| Accuracy | 0.8336 |
| AUC Score | 0.8801 |

### Backhaul Probability — Time-Decay Model (v8.6)

`p_return` is the probability of securing a return (backhaul) load. It uses exponential decay over effective business hours waited at the destination:

```
p_return = e^(-lambda × max(0, T_active - T_grace))

where:
  lambda (FTL)     = 0.062   long-haul trucks: slow decay, higher stability
  lambda (Carting) = 0.15    small vehicles: fast decay, must keep moving
  T_grace          = 4 hours (unloading + paperwork — p_return stays 1.0)
  T_active         = business hours elapsed (08:00–20:00 window)
```

**Vehicle type comparison (same lane, lane_popularity=0.40, 500 km):**

| Vehicle | λ | p_return |
|---|---|---|
| FTL | 0.062 | ~0.69 |
| Carting | 0.150 | ~0.41 |

**Scenarios:**
- Within grace (T_active ≤ 4h): `p_return = 1.0` — no risk cost
- 48h wait (FTL): `p_return ≈ e^(-0.062×44) ≈ 0.066` — ~93% of risk cost applies

Night arrivals (after 20:00) freeze the dwell timer until 08:00 next morning — trucks must rest.

### Stage 2 — Pricing Engine (XGBoost Regressor)

Predicts the **Probabilistic Efficiency Score** (0–1) using the Logic 8.0 formula target:

```
probabilistic_efficiency =
    0.40 × efficiency_score          (operational base)
  + 0.20 × lane_popularity           (demand signal)
  + 0.15 × (1 − risk_score)          (route reliability)
  + 0.15 × (1 − p_delay)             (ML forward-looking)
  + 0.10 × (1 − driver_delay_rate)   (driver reliability)
```

**Results:**

| Metric | Value |
|---|---|
| R² Score | 0.9121 |

### Decision Confidence

A transparency score shown in the UI:

```
data_confidence  = min(1.0, lane_trip_count / 50)
model_confidence = 0.5 + |efficiency_score − 0.5| × 0.8
distance_penalty = 0.08  (if distance was estimated, not haversine)

confidence = (data_confidence × 0.55 + model_confidence × 0.35) − distance_penalty
```

| Confidence | Label | Meaning |
|---|---|---|
| ≥ 0.82 | Elite | 50+ historical trips, strong ML signal |
| ≥ 0.62 | High | Good data and model certainty |
| ≥ 0.42 | Moderate | Some data, reasonable estimate |
| < 0.42 | Low | New lane — treat quote as indicative |

---

## 5-Layer Pricing Engine

### Layer 1 — Operational Base (C_base)

Worst-case baseline — includes **full round-trip** fuel and tolls.

```
F_loaded         = D / M_loaded            (litres, forward trip)
F_empty          = D / M_empty             (litres, return trip)
fuel_cost        = (F_loaded + F_empty) × P_diesel
tolls            = 2 × D × Rs 2.50/km     (round-trip, full tolls)
wages            = (2D / 50 km/h / 24) × Rs 1,200/day
maintenance      = (2D / 50 km/h / 24) × Rs 400/day

C_base = fuel_cost + tolls + wages + maintenance
```

### Layer 2 — Return-Load Risk (C_risk)

Financial exposure of driving back empty — **only the fuel cost**, since full tolls are already in C_base.

```
C_risk = (1 - p_return) × Fuel_Empty_Return
       = (1 - p_return) × (F_empty × P_diesel)

Scenario A (within grace, p_return = 1.0): C_risk = Rs 0
Scenario B (48h wait, p_return ≈ 0.1):    C_risk ≈ 90% of empty-leg fuel
```

### Layer 3 — Pricing Corridor [P_min, P_max]

```
P_min = C_base + C_risk                           (defensive floor)
P_max = P_min
      + lane_popularity × 0.15 × C_base           (demand premium)
      + max(0, seasonal_factor) × C_base           (urgency premium)
```

The corridor gives the transporter negotiation room:
- **P_min** is the minimum to quote without losing money on risk
- **P_max** is what can be asked in high-demand / peak-season conditions

### Layer 4 — Market Snap (Competitor Bounds)

If a competitor price is provided, the corridor is "snapped" to stay within market reality:

```
epsilon_high = 0.12   (+12% ceiling above competitor)
epsilon_low  = 0.10   (-10% floor below competitor)

Floor   = max(P_min, competitor × 0.90)
Ceiling = min(P_max, competitor × 1.12)
```

If no competitor price: floor = P_min, ceiling = P_max (raw corridor).

### Layer 5 — Feasibility Gate + Status

Guarantees the driver covers minimum survival earnings regardless of market conditions. Assigns a trip status.

```
C_survival = F_loaded × P_diesel + D × Rs 2.50 + 0.80 × wages   (one-way survival)

Recommended = max(Floor, C_survival)

margin_pct  = (Recommended - C_survival) / C_base
Status = "Healthy"    if margin_pct >= 0.10
       = "Low Margin" if 0 <= margin_pct < 0.10
       = "Infeasible" if Recommended < C_survival
```

The Status badge appears in the UI route banner — green (Healthy), amber (Low Margin), red (Infeasible).

---

## Model Results

| Model | Metric | Value | Meaning |
|---|---|---|---|
| v7 Regressor | RMSE | 0.0922 | ~9-point error on 0–1 scale |
| v7 Regressor | R² | 0.6044 | 60% variance explained |
| v8 Risk Engine | Accuracy | 0.8336 | 83% correct delay predictions |
| v8 Risk Engine | AUC | 0.8801 | Strong delayed vs on-time discrimination |
| v8 Pricing Engine | R² | 0.9121 | 91% formula approximation accuracy |

### Sample Output (v8.6)

| Route | Weight | Season | p_return | C_base | Recommended |
|---|---|---|---|---|---|
| Delhi → Mumbai | 15T | Winter | 61% | Rs 80,201 | Rs 91,975 |
| Bangalore → Chennai | 10T | Peak | 58% | Rs 30,400 | Rs 36,200 |

---

## API Reference

### `GET /api/health`
```json
{"status": "ok", "version": "v8.6", "models_loaded": true}
```

### `GET /api/cities`
Returns the list of 81 Indian cities known to the distance engine.
```json
{"cities": ["Agra", "Ahmedabad", "Ajmer", ...]}
```

### `POST /api/extract-vision`

Extracts diesel price or competitor freight rate from a photo using Claude Vision.

**Requires:** `ANTHROPIC_API_KEY` environment variable set on the Flask server.

**Request body:**
```json
{
  "image_base64": "<base64-encoded image string>",
  "media_type":   "image/jpeg"
}
```

**Response:**
```json
{
  "authentic_diesel":     87.50,
  "authentic_competitor": 92000,
  "timestamp":            "2026-11-15",
  "source_confidence":    "High",
  "notes":                "HSD diesel rate extracted from IOCL pump board"
}
```

If `ANTHROPIC_API_KEY` is not set, the endpoint returns HTTP 503 with an explanatory message. The frontend gracefully shows an error banner — all other features continue to work normally.

---

### `POST /api/predict`

**Request body:**
```json
{
  "origin":            "Delhi",
  "destination":       "Mumbai",
  "trip_date":         "2026-11-15",
  "weight_tonnes":     15,
  "vehicle_type":      "FTL",
  "competitor_price":  95000,
  "diesel_price":      87.50
}
```

> `vehicle_type`: `"FTL"` (default) or `"CARTING"`. Controls λ decay constant.
> `competitor_price` and `diesel_price` are optional. `diesel_price` is typically provided from the vision extraction result.

**Response:**
```json
{
  "origin": "Delhi",
  "destination": "Mumbai",
  "trip_date": "15 Nov 2026",
  "weight_tonnes": 15.0,

  "abstraction": {
    "distance_km": 1449.5,
    "distance_method": "haversine",
    "day_name": "Sunday",
    "season": "Peak",
    "is_peak_season": true,
    "m_loaded_kmpl": 2.75,
    "m_empty_kmpl": 4.8,
    "lane_popularity": 0.45,
    "route_risk": 0.28,
    "lane_trip_count": 12
  },

  "efficiency_score": 0.6120,
  "efficiency_label": "High",
  "delay_probability": 0.2340,
  "p_return": 0.6090,
  "p_return_model": {
    "arrival_time": "11:00, Sun",
    "expected_dwell_h": 15.0,
    "T_active_h": 12.0,
    "T_grace_h": 4.0,
    "lambda": 0.062,
    "within_grace": false
  },
  "confidence_score": 0.312,
  "confidence_label": "Moderate",

  "pricing": {
    "operational": {
      "fuel_loaded_L": 527.1, "fuel_empty_L": 302.0,
      "fuel_loaded_cost": 46190.5, "fuel_empty_cost": 26462.1,
      "fuel_cost": 72652.6, "tolls_oneway": 3623.8,
      "driver_wage": 2899.0, "maintenance": 966.3,
      "c_base": 80141.7, "m_loaded_kmpl": 2.75
    },
    "risk": {
      "p_return": 0.6090, "p_no_return": 0.3910,
      "empty_fuel_val": 26462.1, "return_tolls": 3623.8,
      "c_risk": 11741.3
    },
    "corridor": {
      "p_min": 91883.0, "p_max": 99476.2,
      "demand_premium": 5409.6, "urgency_premium": 2183.6,
      "floor": 91883.0, "ceiling": 99476.2,
      "was_snapped": false, "competitor_price": null,
      "epsilon_high_pct": 12.0, "epsilon_low_pct": 10.0
    },
    "feasibility": {
      "c_survival": 52018.0,
      "is_feasible": true,
      "recommended": 91883.0,
      "note": "Trip covers driver survival earnings."
    },
    "recommended_price": 91883
  }
}
```

---

## ML Training Scripts

> These scripts are run **once** to generate the trained models. You do not need to re-run them — the `.pkl` files are already included.

### Script 01 — Data Processor (Logic 7.2.5)

```bash
python freightiq_01_data_processor.py
```

Downloads the Kaggle GPS dataset, engineers features, and produces `freightiq_gps_ready_data_v7.csv`. Key engineering:

- **Bayesian Route Risk:** `route_risk = std(delay_flag) × count / (count + 10)` — shrinks toward zero for new routes
- **Distance-Adaptive Time Decay:** `λ = 0.062 / log(1 + dist/100)` — long-haul routes penalised less for dwell time
- **Leakage-Free Driver Score:** `expanding().mean().shift()` — model only sees past performance

### Script 02 — XGBoost v7 Engine

```bash
python freightiq_02_engine_v7.py
```

Trains an XGBoost regressor to approximate the Logic 7.2.5 efficiency score. Uses `RandomizedSearchCV` (15 configs, 3-fold CV). Top feature: `is_deadhead` at 60% importance.

### Script 03 — Elite Stack v8

```bash
python freightiq_03_elite_stack_v8.py
```

Two-stage stacking with data-leakage prevention:

```python
# Out-of-fold predictions prevent Stage 2 from training on inflated Stage 1 outputs
oof_probs = cross_val_predict(
    XGBClassifier(...), X_cls, y_cls,
    cv=StratifiedKFold(5), method='predict_proba'
)[:, 1]
```

---

## Limitations & Caveats

| # | Limitation | Impact |
|---|---|---|
| 1 | **Distance is estimated** — Haversine × 1.32 approximates road distance; actual roads may differ by ±15% | Use Google Distance Matrix API for production accuracy |
| 2 | **Lane confidence is Low for new routes** — the Knowledge Base only has data for routes in the training dataset | Quote is still mathematically sound; treat as indicative for new lanes |
| 3 | **p_return dwell time is estimated** — expected dwell is derived from lane_popularity, not actual booking data | Calibrate expected_dwell formula against real backhaul logs when available |
| 4 | **Diesel price defaults to Rs 87.67/L** — fluctuates daily, but can be overridden by vision scan | Use vision upload feature to pull live pump price from a photo |
| 5 | **v8 R² of 0.91 reflects formula approximation** — the Pricing Engine is trained to replicate Logic 8.0, not validated against real market rates | Validate quote acceptance rates before using in live pricing |
| 6 | **06:00 fixed departure assumption** — arrival time for T_active uses 06:00 departure | Expose departure time as an optional input for tighter T_active estimation |
| 7 | **Vision extraction requires ANTHROPIC_API_KEY** — without it, photo scanning is disabled | All other features work without the key; vision is additive |

---

## Dependencies

### Python
```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `flask` | ≥ 2.3 | REST API server |
| `flask-cors` | ≥ 4.0 | Cross-origin requests from React dev server |
| `xgboost` | ≥ 1.7 | Risk Engine + Pricing Engine |
| `scikit-learn` | ≥ 1.3 | Model selection, metrics, scalers |
| `pandas` | ≥ 2.0 | Data manipulation + Lane KB |
| `numpy` | ≥ 1.24 | Numerical operations |
| `joblib` | ≥ 1.3 | Model serialisation (`.pkl`) |
| `openpyxl` | ≥ 3.1 | Reading `.xlsx` dataset files (Script 01) |
| `kagglehub` | ≥ 0.2 | Dataset download (Script 01) |
| `anthropic` | ≥ 0.20 | Claude Vision for photo extraction (optional) |

### Node / Frontend
```bash
cd frontend && npm install
```

| Package | Purpose |
|---|---|
| `react` + `react-dom` | UI framework |
| `lucide-react` | Icons |
| `vite` | Dev server + bundler |
| `tailwindcss` | Utility CSS |

---

## Dataset

**Source:** [Delivery Truck Trips Data](https://www.kaggle.com/datasets/ramakrishnanthiyagu/delivery-truck-trips-data) — Kaggle

**Size:** 6,880 GPS trip records across multiple vehicles, origins, and destinations in India.
