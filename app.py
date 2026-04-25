"""
FreightIQ Elite v8.6 — Flask REST API
--------------------------------------
Pipeline: Vision Ingestion → Contextual Computation → Corridor Optimisation

User inputs  : origin, destination, trip_date, weight_tonnes, vehicle_type,
               competitor_price, diesel_price (optional — from vision extraction)
System derives: distance, arrival_time, season, M_loaded, lane context
Vision Layer : POST /api/extract-vision  — OCR diesel boards & competitor quotes
ML outputs   : delay_probability (Stage-1), efficiency_score (Stage-2)
p_return     : time-decay  p = e^(-λ × max(0, T_active − T_grace))
               FTL: λ=0.062 / Carting: λ=0.15 / Business window 08:00–20:00
Pricing      : C_base → C_risk → Corridor [P_min, P_max] → Snap → Gate → Status

Endpoints:
    GET  /api/health
    GET  /api/cities              — autocomplete list
    POST /api/extract-vision      — OCR: diesel boards / competitor quotes
    POST /api/predict             — full inference + pricing
"""

from __future__ import annotations   # Python 3.9 compatible type hints

import json
import math
import urllib.request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Optional: Claude vision client ────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _VISION_CLIENT = _anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False


app = Flask(__name__)
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────
risk_engine    = joblib.load("freightiq_risk_engine_v8.pkl")
pricing_engine = joblib.load("freightiq_pricing_engine_v8.pkl")
V8_FEATURES    = joblib.load("v8_feature_list.pkl")
BASE_FEATURES  = [f for f in V8_FEATURES if f != "predicted_delay_prob"]

# ── Physical & financial constants (v8.6) ─────────────────────────────────────
DEFAULT_DIESEL = 87.67    # ₹/L (Uttarakhand region — overridable via vision)
M_EMPTY        = 4.8      # km/L (empty truck)
TOLL_RATE      = 2.50     # ₹/km
AVG_SPEED_KMPH = 50
DRIVER_DAILY   = 1_200    # ₹/day
MAINT_DAILY    = 400      # ₹/day
ROAD_FACTOR    = 1.32
EPSILON_HIGH   = 0.12     # +12% competitor ceiling
EPSILON_LOW    = 0.10     # −10% competitor floor

# ── Time-decay constants (v8.6) ───────────────────────────────────────────────
T_GRACE        = 4.0      # grace period (unloading + paperwork), hours
LAMBDA_FTL     = 0.062    # Full Truck Load — slow decay (long-haul stability)
LAMBDA_CARTING = 0.15     # Carting / small vehicle — fast decay (short-haul)
BUSINESS_START = 8        # 08:00
BUSINESS_END   = 20       # 20:00
BUSINESS_HOURS = 12       # active hours per day

# ── Vision extraction prompt ──────────────────────────────────────────────────
VISION_PROMPT = """SYSTEM ROLE: Logistics Data Authenticator (FreightIQ Elite v8.6)

You are the primary input sensor for FreightIQ. You will receive an image containing logistics market data (diesel rate boards, WhatsApp freight quotes, or rate charts).

EXTRACTION OBJECTIVES:
1. Diesel Rate (authentic_diesel): Locate the per-litre price. Keywords: Diesel, HSD, Rs/L. If multiple prices exist (e.g., Cash vs. Credit), extract the CASH price.
2. Competitor Rate (authentic_competitor): Identify the quoted freight price for the specific route mentioned.
3. Validation Metadata: Identify the date and time visible in the image.
4. Source Type: Classify the image as "fuel_board", "chat_screenshot", or "rate_chart".
5. Confidence: Assign "High" if numbers are clearly legible; "Low" if blurry, ambiguous, or unclear.

Return ONLY a valid JSON object in this exact format (use null for fields not found):
{
  "authentic_diesel":     <float or null>,
  "authentic_competitor": <float or null>,
  "timestamp":            "<ISO-8601 date string or null>",
  "source_type":          "fuel_board" | "chat_screenshot" | "rate_chart",
  "confidence_level":     "High" | "Low",
  "notes":                "<one-line description of what was found>"
}"""

# ── Indian city coordinates (lat, lon) ────────────────────────────────────────
CITY_COORDS = {
    "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777), "bombay": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639), "calcutta": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707), "madras": (13.0827, 80.2707),
    "bangalore": (12.9716, 77.5946), "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "nagpur": (21.1458, 79.0882),
    "indore": (22.7196, 75.8577),
    "bhopal": (23.2599, 77.4126),
    "patna": (25.5941, 85.1376),
    "surat": (21.1702, 72.8311),
    "vadodara": (22.3072, 73.1812),
    "ludhiana": (30.9010, 75.8573),
    "amritsar": (31.6340, 74.8723),
    "chandigarh": (30.7333, 76.7794),
    "kochi": (9.9312, 76.2673),
    "coimbatore": (11.0168, 76.9558),
    "visakhapatnam": (17.6868, 83.2185), "vizag": (17.6868, 83.2185),
    "bhubaneswar": (20.2961, 85.8245),
    "guwahati": (26.1445, 91.7362),
    "roorkee": (29.8543, 77.8880),
    "haridwar": (29.9457, 78.1642),
    "dehradun": (30.3165, 78.0322),
    "meerut": (28.9845, 77.7064),
    "agra": (27.1767, 78.0081),
    "varanasi": (25.3176, 82.9739),
    "prayagraj": (25.4358, 81.8463), "allahabad": (25.4358, 81.8463),
    "jodhpur": (26.2389, 73.0243),
    "udaipur": (24.5854, 73.7125),
    "kota": (25.2138, 75.8648),
    "gwalior": (26.2183, 78.1828),
    "jabalpur": (23.1815, 79.9864),
    "raipur": (21.2514, 81.6296),
    "ranchi": (23.3441, 85.3096),
    "jamshedpur": (22.8046, 86.2029),
    "jammu": (32.7266, 74.8570),
    "goa": (15.2993, 74.1240),
    "panaji": (15.4909, 73.8278),
    "thiruvananthapuram": (8.5241, 76.9366),
    "mangalore": (12.8703, 74.8421),
    "mysore": (12.2958, 76.6394), "mysuru": (12.2958, 76.6394),
    "hubli": (15.3647, 75.1240),
    "vijayawada": (16.5062, 80.6480),
    "madurai": (9.9252, 78.1198),
    "rajkot": (22.3039, 70.8022),
    "bikaner": (28.0229, 73.3119),
    "ajmer": (26.4499, 74.6399),
    "siliguri": (26.7271, 88.3953),
    "faridabad": (28.4089, 77.3178),
    "gurgaon": (28.4595, 77.0266), "gurugram": (28.4595, 77.0266),
    "noida": (28.5355, 77.3910),
    "thane": (19.2183, 72.9781),
    "nashik": (19.9975, 73.7898),
    "aurangabad": (19.8762, 75.3433),
    "solapur": (17.6599, 75.9064),
    "kanchipuram": (12.8185, 79.6947),
    "pondicherry": (11.9416, 79.8083), "puducherry": (11.9416, 79.8083),
    "salem": (11.6643, 78.1460),
    "trichy": (10.7905, 78.7047), "tiruchirappalli": (10.7905, 78.7047),
    "medak": (18.0490, 78.2630),
    "nadia": (23.4700, 88.5500),
    "raigarh": (21.8974, 83.3950),
    "anekal": (12.7100, 77.6970),
    "guntur": (16.2991, 80.4575),
    "nellore": (14.4426, 79.9865),
}

KNOWN_CITIES = sorted({c.title() for c in CITY_COORDS})

# ── Lane Knowledge Base — built from CSV ──────────────────────────────────────
def _build_lane_kb() -> tuple[dict, dict]:
    try:
        df = pd.read_csv("freightiq_gps_ready_data_v7.csv")

        def city_key(loc: str) -> str:
            parts = [p.strip().lower() for p in str(loc).split(",")]
            for token in parts[1:]:
                token = token.strip()
                if token and token not in ("india", ""):
                    return token
            return parts[0].strip()

        df["_ok"] = df["Origin_Location"].apply(city_key)
        df["_dk"] = df["Destination_Location"].apply(city_key)

        kb = {}
        for (ok, dk), grp in df.groupby(["_ok", "_dk"]):
            kb[(ok, dk)] = {
                "lane_popularity":   float(grp["lane_popularity"].mean()),
                "route_risk":        float(grp["route_risk"].mean()),
                "driver_delay_rate": float(grp["driver_delay_rate"].mean()),
                "trip_count":        len(grp),
                "avg_dist_km":       float(grp["TRANSPORTATION_DISTANCE_IN_KM"].mean()),
            }

        g_avg = {
            "lane_popularity":   float(df["lane_popularity"].mean()),
            "route_risk":        float(df["route_risk"].mean()),
            "driver_delay_rate": float(df["driver_delay_rate"].mean()),
            "trip_count":        0,
            "avg_dist_km":       float(df["TRANSPORTATION_DISTANCE_IN_KM"].mean()),
        }
        return kb, g_avg
    except Exception as e:
        print(f"Lane KB build failed: {e}")
        return {}, {"lane_popularity": 0.35, "route_risk": 0.30,
                    "driver_delay_rate": 0.20, "trip_count": 0, "avg_dist_km": 500}

LANE_KB, GLOBAL_AVG = _build_lane_kb()


def lookup_lane(origin: str, destination: str) -> dict:
    ok = origin.strip().lower()
    dk = destination.strip().lower()
    if (ok, dk) in LANE_KB:
        return LANE_KB[(ok, dk)]
    for (k_ok, k_dk), v in LANE_KB.items():
        if ok in k_ok and dk in k_dk:
            return v
    return dict(GLOBAL_AVG)


# ── Distance engine ───────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def osrm_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
    """Real road distance via OSRM public API (OpenStreetMap). Free, no key required."""
    # OSRM coordinate order: longitude first
    url = (f"http://router.project-osrm.org/route/v1/driving/"
           f"{lon1},{lat1};{lon2},{lat2}?overview=false")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FreightIQ/8.6"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        if data.get("code") == "Ok":
            return data["routes"][0]["distance"] / 1000   # metres → km
    except Exception:
        pass
    return None


def calc_distance(origin: str, destination: str, lane_avg_km: float,
                  origin_coords: dict | None = None,
                  dest_coords:   dict | None = None) -> tuple[float, str]:
    # Tier 1: OSRM with coordinates from frontend (Nominatim-selected, any city)
    if origin_coords and dest_coords:
        d = osrm_distance(origin_coords["lat"], origin_coords["lon"],
                          dest_coords["lat"],   dest_coords["lon"])
        if d:
            return round(d, 1), "osrm"

    # Tier 2: OSRM with known city coordinates (81-city dict)
    o  = origin.strip().lower()
    dk = destination.strip().lower()
    if o in CITY_COORDS and dk in CITY_COORDS:
        d = osrm_distance(*CITY_COORDS[o], *CITY_COORDS[dk])
        if d:
            return round(d, 1), "osrm"
        # Haversine if OSRM is temporarily unreachable
        straight = haversine_km(*CITY_COORDS[o], *CITY_COORDS[dk])
        return round(straight * ROAD_FACTOR, 1), "haversine"

    # Tier 3: historical lane average
    if lane_avg_km > 0:
        return round(lane_avg_km, 1), "historical_avg"

    return 500.0, "default_fallback"


# ── Temporal feature extraction ───────────────────────────────────────────────
SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Pre-Summer",
    4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Peak", 11: "Peak", 12: "Peak",
}

SEASON_DEMAND = {
    "Peak": 0.18, "Winter": 0.05, "Pre-Summer": 0.0,
    "Summer": -0.05, "Monsoon": -0.10,
}


def extract_temporal(date_str: str) -> dict:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "day_of_week": dt.weekday(),
        "month":       dt.month,
        "season":      SEASON_MAP[dt.month],
        "is_peak":     dt.month in (10, 11, 12),
        "day_name":    dt.strftime("%A"),
        "formatted":   dt.strftime("%d %b %Y"),
        "dt":          dt,
    }


# ── Physics engine ────────────────────────────────────────────────────────────
def loaded_mileage(weight_tonnes: float) -> float:
    """M_loaded = 3.5 − 0.02 × weight  (min 1.5 km/L)"""
    return max(1.5, 3.5 - 0.02 * weight_tonnes)


# ── p_return — Time-Decay Model (v8.6) ───────────────────────────────────────
def _active_hours(arrival: datetime, dwell_hours: float) -> float:
    """
    Business-hours-adjusted dwell time.
    Window: 08:00–20:00 (12 active hours/day).
    Night arrivals defer the clock until next morning.
    """
    h = arrival.hour + arrival.minute / 60.0

    if h < BUSINESS_START:
        skip = BUSINESS_START - h
        effective = max(0.0, dwell_hours - skip)
        remaining_today = float(BUSINESS_HOURS)
    elif h >= BUSINESS_END:
        skip = (24.0 - h) + BUSINESS_START
        effective = max(0.0, dwell_hours - skip)
        remaining_today = float(BUSINESS_HOURS)
    else:
        effective = dwell_hours
        remaining_today = BUSINESS_END - h

    if effective <= remaining_today:
        return effective
    overflow = effective - remaining_today
    return remaining_today + overflow * (BUSINESS_HOURS / 24.0)


def compute_p_return(trip_date: str, distance: float,
                     lane_popularity: float,
                     vehicle_type: str = "FTL") -> tuple[float, dict]:
    """
    p_return = e^(-λ × max(0, T_active − T_grace))

    λ: 0.062 for FTL (long-haul stability) / 0.15 for Carting (short-haul speed)
    T_grace: 4 h grace period — p_return = 1.0 during this window.
    T_active: business hours elapsed during expected dwell at destination.
    Expected dwell estimated from lane popularity.
    Assumes 06:00 departure.
    """
    lam = LAMBDA_FTL if vehicle_type.upper() != "CARTING" else LAMBDA_CARTING

    dt        = datetime.strptime(trip_date, "%Y-%m-%d")
    departure = dt.replace(hour=6, minute=0, second=0)
    travel_h  = distance / AVG_SPEED_KMPH
    arrival   = departure + timedelta(hours=travel_h)

    expected_dwell_h = T_GRACE + (1.0 - lane_popularity) * 20.0
    T_active = _active_hours(arrival, expected_dwell_h)
    p = math.exp(-lam * max(0.0, T_active - T_GRACE))
    p = float(np.clip(p, 0.05, 0.99))

    return p, {
        "arrival_time":     arrival.strftime("%H:%M, %a"),
        "expected_dwell_h": round(expected_dwell_h, 1),
        "T_active_h":       round(T_active, 1),
        "T_grace_h":        T_GRACE,
        "lambda":           lam,
        "vehicle_type":     vehicle_type.upper(),
        "within_grace":     T_active <= T_GRACE,
    }


# ── Decision confidence ───────────────────────────────────────────────────────
def decision_confidence(trip_count: int, efficiency_score: float,
                        dist_method: str) -> tuple[float, str]:
    data_conf  = min(1.0, trip_count / 50)
    model_conf = 0.5 + abs(efficiency_score - 0.5) * 0.8
    dist_pen   = 0.0 if dist_method in ("historical_avg", "osrm") else 0.08
    score = (data_conf * 0.55 + model_conf * 0.35) - dist_pen
    score = float(np.clip(score, 0.10, 0.97))
    label = ("Elite"    if score >= 0.82 else
             "High"     if score >= 0.62 else
             "Moderate" if score >= 0.42 else "Low")
    return round(score, 3), label


# ── Trip status ───────────────────────────────────────────────────────────────
def determine_status(recommended_price: float, c_base: float,
                     c_survival: float) -> str:
    """Healthy / Low Margin / Infeasible based on margin above survival floor."""
    if recommended_price < c_survival:
        return "Infeasible"
    margin_pct = (recommended_price - c_survival) / c_base if c_base > 0 else 0
    return "Healthy" if margin_pct >= 0.10 else "Low Margin"


# ── 5-Layer Pricing Engine (v8.6) ─────────────────────────────────────────────
def pricing_v86(
    distance: float, weight: float, p_return: float,
    lane_popularity: float, route_risk: float, season: str,
    competitor_price: float | None,
    diesel_price: float = DEFAULT_DIESEL,
) -> dict:

    m_loaded    = loaded_mileage(weight)
    round_hours = (distance * 2.0) / AVG_SPEED_KMPH   # A→B→A total hours

    # ── Layer 1: Operational Base (C_base) ────────────────────────────────────
    # "full tolls" = round-trip: both legs covered.
    F_loaded         = distance / m_loaded
    F_empty          = distance / M_EMPTY
    fuel_loaded_cost = F_loaded * diesel_price
    fuel_empty_cost  = F_empty  * diesel_price
    fuel_cost        = fuel_loaded_cost + fuel_empty_cost
    tolls            = 2.0 * distance * TOLL_RATE       # round-trip tolls
    wages            = (round_hours / 24.0) * DRIVER_DAILY
    maintenance      = (round_hours / 24.0) * MAINT_DAILY
    c_base           = fuel_cost + tolls + wages + maintenance

    operational = {
        "fuel_loaded_L":    round(F_loaded, 2),
        "fuel_empty_L":     round(F_empty, 2),
        "fuel_loaded_cost": round(fuel_loaded_cost, 2),
        "fuel_empty_cost":  round(fuel_empty_cost, 2),
        "fuel_cost":        round(fuel_cost, 2),
        "tolls_roundtrip":  round(tolls, 2),
        "driver_wage":      round(wages, 2),
        "maintenance":      round(maintenance, 2),
        "c_base":           round(c_base, 2),
        "m_loaded_kmpl":    round(m_loaded, 2),
        "diesel_price":     round(diesel_price, 2),
    }

    # ── Layer 2: Return-Load Risk (C_risk) ────────────────────────────────────
    # C_risk = (1 − p_return) × Fuel_Empty_Return
    c_risk = (1.0 - p_return) * fuel_empty_cost

    risk_data = {
        "p_return":       round(p_return, 4),
        "p_no_return":    round(1.0 - p_return, 4),
        "empty_fuel_val": round(fuel_empty_cost, 2),
        "c_risk":         round(c_risk, 2),
    }

    # ── Layer 3: Pricing Corridor ─────────────────────────────────────────────
    # P_min = C_base + C_risk  (defensive floor, covers all guaranteed costs)
    # P_max = P_min × 1.15     (flat 15% demand+urgency premium for negotiation headroom)
    p_min    = c_base + c_risk
    p_max    = p_min * 1.15
    premium  = p_max - p_min   # = P_min × 0.15

    # ── Layer 4: Competitive Snap ─────────────────────────────────────────────
    # Anchor the corridor to authenticated competitor price: [-10%, +12%]
    floor   = p_min
    ceiling = p_max
    was_snapped = False
    if competitor_price and competitor_price > 0:
        snap_floor   = competitor_price * (1.0 - EPSILON_LOW)
        snap_ceiling = competitor_price * (1.0 + EPSILON_HIGH)
        floor        = max(p_min, snap_floor)
        ceiling      = min(p_max, snap_ceiling)
        was_snapped  = True
        if floor > ceiling:
            ceiling = floor

    corridor = {
        "p_min":            round(p_min, 2),
        "p_max":            round(p_max, 2),
        "premium":          round(premium, 2),       # the 15% demand/urgency headroom
        "floor":            round(floor, 2),
        "ceiling":          round(ceiling, 2),
        "was_snapped":      was_snapped,
        "competitor_price": round(competitor_price, 2) if competitor_price else None,
        "epsilon_high_pct": EPSILON_HIGH * 100,
        "epsilon_low_pct":  EPSILON_LOW * 100,
    }

    # ── Layer 5: Feasibility Gate ─────────────────────────────────────────────
    # C_survival = Fuel_Loaded + one-way Tolls + 80% × Wages
    c_survival  = fuel_loaded_cost + (distance * TOLL_RATE) + 0.80 * wages
    recommended = floor
    is_feasible = recommended >= c_survival
    if not is_feasible:
        recommended = c_survival

    status = determine_status(recommended, c_base, c_survival)

    feasibility = {
        "c_survival":  round(c_survival, 2),
        "is_feasible": is_feasible,
        "recommended": round(recommended, 2),
        "status":      status,
        "note": (
            "Trip covers driver survival earnings."
            if is_feasible
            else f"Price raised to survival floor Rs {c_survival:,.0f} — review route viability."
        ),
    }

    return {
        "operational":       operational,
        "risk":              risk_data,
        "corridor":          corridor,
        "feasibility":       feasibility,
        "recommended_price": round(recommended, 2),
        "status":            status,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "version":          "v8.6",
        "models_loaded":    True,
        "vision_available": VISION_AVAILABLE,
        "distance_engine":  "OSRM (OpenStreetMap)",
        "autocomplete":     "Nominatim (OpenStreetMap)",
    })


@app.route("/api/cities", methods=["GET"])
def cities():
    return jsonify({"cities": KNOWN_CITIES})


@app.route("/api/extract-vision", methods=["POST"])
def extract_vision():
    """
    Stage A — Vision & Authentication Layer.
    Accepts a base64-encoded image and returns extracted diesel / competitor price.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    if not VISION_AVAILABLE:
        return jsonify({
            "error": "Vision extraction unavailable. Set the ANTHROPIC_API_KEY environment variable.",
            "authentic_diesel": None,
            "authentic_competitor": None,
        }), 503

    data       = request.get_json(force=True)
    image_b64  = data.get("image_base64", "")
    media_type = data.get("media_type", "image/jpeg")

    if not image_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        msg = _VISION_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       image_b64,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )

        raw = msg.content[0].text.strip()
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            extracted = json.loads(raw[start:end])
        else:
            extracted = {
                "authentic_diesel":     None,
                "authentic_competitor": None,
                "timestamp":            None,
                "source_confidence":    "Low",
                "notes":                "Could not parse structured response from image.",
            }

        return jsonify(extracted)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Empty or invalid JSON body"}), 400

        required = ["origin", "destination", "trip_date", "weight_tonnes"]
        missing  = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        origin        = str(data["origin"]).strip()
        destination   = str(data["destination"]).strip()
        weight        = float(data["weight_tonnes"])
        vehicle_type  = str(data.get("vehicle_type", "FTL")).strip().upper()
        comp_price    = float(data["competitor_price"]) if data.get("competitor_price") else None
        diesel_price  = float(data["diesel_price"]) if data.get("diesel_price") else DEFAULT_DIESEL

        if vehicle_type not in ("FTL", "CARTING"):
            vehicle_type = "FTL"

        # ── Intelligence Abstraction Layer ─────────────────────────────────────
        temporal = extract_temporal(data["trip_date"])
        lane_ctx = lookup_lane(origin, destination)

        # Optional coords from frontend Nominatim selection (enables OSRM for any city)
        origin_coords = ({"lat": float(data["origin_lat"]), "lon": float(data["origin_lon"])}
                         if data.get("origin_lat") and data.get("origin_lon") else None)
        dest_coords   = ({"lat": float(data["dest_lat"]),   "lon": float(data["dest_lon"])}
                         if data.get("dest_lat") and data.get("dest_lon") else None)

        distance, dist_method = calc_distance(
            origin, destination, lane_ctx["avg_dist_km"], origin_coords, dest_coords
        )
        m_loaded = loaded_mileage(weight)

        abstraction = {
            "distance_km":     distance,
            "distance_method": dist_method,
            "day_name":        temporal["day_name"],
            "season":          temporal["season"],
            "is_peak_season":  temporal["is_peak"],
            "m_loaded_kmpl":   round(m_loaded, 2),
            "m_empty_kmpl":    M_EMPTY,
            "lane_popularity": round(lane_ctx["lane_popularity"], 3),
            "route_risk":      round(lane_ctx["route_risk"], 3),
            "lane_trip_count": lane_ctx["trip_count"],
            "vehicle_type":    vehicle_type,
            "diesel_price":    round(diesel_price, 2),
            "diesel_source":   "vision" if data.get("diesel_price") else "default",
        }

        # ── ML Inference ───────────────────────────────────────────────────────
        trip = pd.DataFrame([{
            "TRANSPORTATION_DISTANCE_IN_KM": distance,
            "day_of_week":       temporal["day_of_week"],
            "month":             temporal["month"],
            "lane_popularity":   lane_ctx["lane_popularity"],
            "route_risk":        lane_ctx["route_risk"],
            "driver_delay_rate": lane_ctx["driver_delay_rate"],
            "is_return_trip":    0,
            "is_deadhead":       0,
        }])

        p_delay   = float(risk_engine.predict_proba(trip[BASE_FEATURES])[0, 1])
        trip["predicted_delay_prob"] = p_delay
        eff_score = float(np.clip(pricing_engine.predict(trip[V8_FEATURES])[0], 0, 1))

        # ── p_return: Time-Decay Model ─────────────────────────────────────────
        p_return, p_return_model = compute_p_return(
            data["trip_date"], distance, lane_ctx["lane_popularity"], vehicle_type
        )

        confidence_score, confidence_label = decision_confidence(
            lane_ctx["trip_count"], eff_score, dist_method
        )

        eff_label = (
            "Elite"  if eff_score >= 0.85 else
            "High"   if eff_score >= 0.65 else
            "Medium" if eff_score >= 0.35 else "Low"
        )

        # ── Pricing (v8.6) ────────────────────────────────────────────────────
        pricing = pricing_v86(
            distance, weight, p_return,
            lane_ctx["lane_popularity"], lane_ctx["route_risk"],
            temporal["season"], comp_price, diesel_price,
        )

        return jsonify({
            "origin":            origin,
            "destination":       destination,
            "trip_date":         temporal["formatted"],
            "weight_tonnes":     weight,
            "vehicle_type":      vehicle_type,
            "abstraction":       abstraction,
            "efficiency_score":  round(eff_score, 4),
            "efficiency_label":  eff_label,
            "delay_probability": round(p_delay, 4),
            "p_return":          round(p_return, 4),
            "p_return_model":    p_return_model,
            "confidence_score":  confidence_score,
            "confidence_label":  confidence_label,
            "pricing":           pricing,
            "status":            pricing["status"],
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("FreightIQ Elite v8.6 — API running at http://localhost:5000")
    if not VISION_AVAILABLE:
        print("  Vision:     DISABLED (set ANTHROPIC_API_KEY to enable)")
    else:
        print("  Vision:     ENABLED  (Claude claude-sonnet-4-6)")
    print("  Distance:   OSRM via OpenStreetMap (free, no key)")
    print("  Autocomplete: Nominatim via OpenStreetMap (free, no key)")
    app.run(debug=True, port=5000)
