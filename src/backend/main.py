from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model  = joblib.load("../../predictions/gb_model.pkl")
pca    = joblib.load("../../predictions/pca_transformer.pkl")
scaler = joblib.load("../../predictions/data_scaler.pkl")

T_MAX_TRAIN = 85.0
H_MAX_TRAIN = 95.0
D_MAX_TRAIN = 75.0

CITIES = {
    "Boston":      {"lat": 42.36, "lon": -71.06},
    "Miami":       {"lat": 25.77, "lon": -80.19},
    "Chicago":     {"lat": 41.88, "lon": -87.63},
    "Dallas":      {"lat": 32.78, "lon": -96.80},
    "Denver":      {"lat": 39.74, "lon": -104.98},
    "Phoenix":     {"lat": 33.45, "lon": -112.07},
    "Seattle":     {"lat": 47.61, "lon": -122.33},
    "Los Angeles": {"lat": 34.05, "lon": -118.24},
}

class LocationRequest(BaseModel):
    city: str

def get_nearest_grid_point(lat, lon, headers):
    """Try the exact point, then spiral outward until NWS finds a valid grid."""
    offsets = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    for dlat in offsets:
        for dlon in offsets:
            try:
                res = requests.get(
                    f"https://api.weather.gov/points/{lat+dlat},{lon+dlon}",
                    headers=headers
                ).json()
                if "properties" in res and "observationStations" in res["properties"]:
                    return res["properties"]["observationStations"], dlat, dlon
            except:
                continue
    raise Exception("No NWS station found near this location")

def get_live_prediction(lat, lon):
    headers = {"User-Agent": "DS3010_Final"}

    point_res = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers).json()
    obs_url   = point_res["properties"]["observationStations"]
    stations  = requests.get(obs_url, headers=headers).json()
    station   = stations["features"][0]["id"]

    obs_url, dlat, dlon = get_nearest_grid_point(lat, lon, headers)
    stations  = requests.get(obs_url, headers=headers).json()
    station   = stations["features"][0]["id"]

    response  = requests.get(f"{station}/observations?limit=500", headers=headers)
    data      = response.json()

    records = []
    for p in data["features"]:
        prop      = p.get("properties", {})
        temp_val  = prop.get("temperature", {}).get("value")
        dew_val   = prop.get("dewpoint", {}).get("value")
        hum_val   = prop.get("relativeHumidity", {}).get("value")
        precip    = prop.get("precipitationLastHour", {}).get("value") or 0
        wind      = prop.get("windSpeed", {}).get("value") or 0
        if temp_val is not None and dew_val is not None:
            records.append({
                "timestamp": prop.get("timestamp"),
                "temp":      (temp_val * 9/5) + 32,
                "humidity":  hum_val,
                "dew_point": (dew_val * 9/5) + 32,
                "precip":    precip,
                "wind_speed": wind,
            })

    df = pd.DataFrame(records).dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Temp_Change_Velocity"] = df["temp"].diff()
    df["In_Risk_Window"] = (
        (df["temp"].between(15, 35)) & (df["dew_point"].between(10, 30))
    ).astype(int)

    df_weekly = df.resample("W", on="timestamp").mean().sort_index()
    df_weekly = df_weekly.rename(columns={
        "temp":       "avg_temp_prior_week",
        "humidity":   "avg_humidity_prior_week",
        "dew_point":  "avg_dew_point_prior_week",
        "precip":     "avg_precip_prior_week",
        "wind_speed": "avg_wind_speed_prior_week",
    })
    df_weekly["temp_stress"] = T_MAX_TRAIN - df_weekly["avg_temp_prior_week"]
    df_weekly["hum_stress"]  = H_MAX_TRAIN - df_weekly["avg_humidity_prior_week"]
    df_weekly["dew_stress"]  = D_MAX_TRAIN - df_weekly["avg_dew_point_prior_week"]
    df_weekly["RSI"]         = (df_weekly["temp_stress"] + df_weekly["hum_stress"] + df_weekly["dew_stress"]) / 3
    df_weekly["RSI_Sustained"]  = df_weekly["RSI"].rolling(4, min_periods=1).mean()
    df_weekly["RSI_Volatility"] = df_weekly["RSI"].rolling(3, min_periods=1).std().fillna(0)
    df_weekly = df_weekly.ffill().bfill().fillna(0)

    latest = df_weekly.iloc[[-1]].fillna(0)

    pca_input_vars = ["RSI", "RSI_Sustained", "RSI_Volatility", "avg_precip_prior_week", "avg_wind_speed_prior_week"]
    pca_out  = pca.transform(scaler.transform(latest[pca_input_vars]))
    raw_vals = latest[["Temp_Change_Velocity", "In_Risk_Window"]].values
    X        = np.hstack([pca_out, raw_vals])

    pca_cols = [f"PC{i+1}" for i in range(pca_out.shape[1])]
    X_df     = pd.DataFrame(X, columns=pca_cols + ["Temp_Change_Velocity", "In_Risk_Window"])

    prob = np.expm1(model.predict(X_df))[0]
    breaks = [0, 4.67898e-05, 0.000154293, 0.0002945734, 0.0008872]
    labels = ["Low", "Moderate", "High", "Critical"]
    category = pd.cut([prob], bins=breaks, labels=labels)[0]

    return {"score": round(float(prob), 8), "category": str(category)}

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("../frontend/index.html") as f:
        return f.read()

@app.get("/cities")
def get_cities():
    return list(CITIES.keys())

class LocationRequest(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict(req: LocationRequest):
    result = get_live_prediction(req.lat, req.lon)
    return result

# @app.post("/predict")
# def predict(req: LocationRequest):
#     if req.city not in CITIES:
#         return {"error": "Unknown city"}
#     coords = CITIES[req.city]
#     result = get_live_prediction(coords["lat"], coords["lon"])
#     return {"city": req.city, **result}