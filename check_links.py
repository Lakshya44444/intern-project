import ast, sys

errors = []

# ── 1. Backend ────────────────────────────────────────────────────────────────
with open("e:/final_intern/app.py", encoding="utf-8") as f:
    be = f.read()

try:
    ast.parse(be)
    print("PASS  app.py syntax")
except SyntaxError as e:
    print("FAIL  app.py syntax:", e)
    sys.exit(1)

# FastAPI: fields are declared in PredictRequest / body.field — check both Pydantic model and usage
reads = [
    "origin", "destination", "trip_date", "weight_tonnes", "vehicle_type",
    "competitor_price", "diesel_price", "addons",
    "origin_lat", "origin_lon", "dest_lat", "dest_lon",
]
for k in reads:
    # Pydantic field declaration OR body.field access
    ok = (f"    {k}:" in be) or (f"body.{k}" in be)
    print(("PASS" if ok else "FAIL") + "  backend reads: " + k)

# Fields backend returns — look for quoted key in return dict or JSONResponse
ret_section = be[be.index("return {"):]
returns = [
    "origin", "destination", "trip_date", "weight_tonnes", "vehicle_type",
    "abstraction", "efficiency_score", "efficiency_label", "delay_probability",
    "p_return", "p_return_model", "confidence_score", "confidence_label",
    "pricing", "addons", "final_price", "status",
]
for k in returns:
    ok = ('"' + k + '"') in ret_section
    print(("PASS" if ok else "FAIL") + "  backend returns: " + k)

# Stale globals that must be gone
stale = ["TOLL_RATE", "DRIVER_DAILY", "MAINT_DAILY", "LAMBDA_FTL", "LAMBDA_CARTING",
         "MOBILIZATION_COST", "m_loaded = loaded_mileage(weight)"]
for n in stale:
    print(("PASS" if n not in be else "FAIL  STALE: ") + "  removed: " + n)

# Required symbols
required = ["ADDON_CONFIG", "VEHICLE_CONFIG", "VALID_VEHICLE_TYPES", "api/autocomplete",
            "FastAPI", "CORSMiddleware", "PredictRequest", "uvicorn"]
for n in required:
    print(("PASS" if n in be else "FAIL") + "  present: " + n)

# ── 2. InputForm.jsx ──────────────────────────────────────────────────────────
with open("e:/final_intern/frontend/src/components/InputForm.jsx", encoding="utf-8") as f:
    fe_input = f.read()

print()
print("--- InputForm.jsx ---")

sent = [
    "weight_tonnes", "competitor_price", "vehicle_type", "diesel_price", "addons",
    "origin_lat", "origin_lon", "dest_lat", "dest_lon",
]
for k in sent:
    print(("PASS" if k in fe_input else "FAIL") + "  sends: " + k)

addon_ids = ["helper_required", "extra_tarpaulin", "extra_rope", "owner_escort", "express_delivery"]
for a in addon_ids:
    ok = a in fe_input and a in be
    print(("PASS" if ok else "FAIL") + "  addon id synced: " + a)

vt_ids = ["LCV", "MCV", "FTL", "CARTING"]
for v in vt_ids:
    ok = ("'" + v + "'") in fe_input and ('"' + v + '"') in be
    print(("PASS" if ok else "FAIL") + "  vehicle type synced: " + v)

# ── 3. ResultDashboard.jsx ────────────────────────────────────────────────────
with open("e:/final_intern/frontend/src/components/ResultDashboard.jsx", encoding="utf-8") as f:
    fe_result = f.read()

print()
print("--- ResultDashboard.jsx ---")

consumed = [
    "origin", "destination", "trip_date", "weight_tonnes",
    "abstraction", "efficiency_score", "efficiency_label",
    "delay_probability", "p_return", "p_return_model",
    "confidence_score", "confidence_label",
    "pricing", "addons", "final_price",
]
for k in consumed:
    print(("PASS" if k in fe_result else "FAIL") + "  consumes: " + k)

sub = [
    "pricing.operational", "pricing.risk", "pricing.corridor", "pricing.feasibility",
    "pricing.recommended_price",
    "addons.selected", "addons.breakdown", "addons.total",
    "abstraction.distance_km", "abstraction.vehicle_type",
    "abstraction.m_loaded_kmpl", "abstraction.m_empty_kmpl",
]
for s in sub:
    print(("PASS" if s in fe_result else "FAIL") + "  accesses: " + s)

print()
print("Done.")
