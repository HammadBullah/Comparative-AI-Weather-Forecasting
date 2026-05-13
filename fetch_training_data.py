"""
fetch_training_data.py
----------------------
Downloads historical daily weather data from Open-Meteo (free, no API key)
for one representative city per climate zone.

Run once before training:
    python fetch_training_data.py

Output: data/<zone>.csv
"""

import os
import requests
import pandas as pd

os.makedirs("data", exist_ok=True)

# ── Representative cities per climate zone ────────────────────────────────────
ZONES = {
    "tropical":    {"city": "Singapore",  "lat":  1.29,  "lon": 103.85},
    "arid":        {"city": "Dubai",      "lat": 25.20,  "lon":  55.27},
    "temperate":   {"city": "London",     "lat": 51.51,  "lon":  -0.13},
    "continental": {"city": "Chicago",    "lat": 41.85,  "lon": -87.65},
    "polar":       {"city": "Reykjavik",  "lat": 64.13,  "lon": -21.94},
}

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

VARIABLES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
]


def fetch_zone(zone: str, info: dict) -> pd.DataFrame:
    print(f"Fetching {zone} ({info['city']})...")

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   info["lat"],
        "longitude":  info["lon"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(VARIABLES),
        "timezone":   "UTC",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "DATE"

    # Rename columns to short names
    df = df.rename(columns={
        "temperature_2m_mean":         "TG",
        "precipitation_sum":           "RR",
        "windspeed_10m_max":           "WS",
        "relative_humidity_2m_mean":   "HU",
        "pressure_msl_mean":           "PP",
    })

    df = df.interpolate()
    return df


if __name__ == "__main__":
    for zone, info in ZONES.items():
        try:
            df = fetch_zone(zone, info)
            path = f"data/{zone}.csv"
            df.to_csv(path)
            print(f"  ✅ Saved {path}  ({len(df)} days)\n")
        except Exception as e:
            print(f"  ❌ Failed {zone}: {e}\n")

    print("Done. All zone data saved to data/")
