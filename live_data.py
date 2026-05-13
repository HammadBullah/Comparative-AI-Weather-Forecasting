"""
live_data.py
------------
Fetches recent 30-day weather history + city name for any lat/lon
using Open-Meteo's free archive API and Open-Meteo geocoding.

No API key required.
"""

import requests
import pandas as pd
from datetime import date, timedelta


VARIABLES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
]

COL_MAP = {
    "temperature_2m_mean":       "TG",
    "precipitation_sum":         "RR",
    "windspeed_10m_max":         "WS",
    "relative_humidity_2m_mean": "HU",
    "pressure_msl_mean":         "PP",
}


def get_recent_weather(lat: float, lon: float, days: int = 60) -> pd.DataFrame:
    """
    Fetch the last `days` days of daily weather for a given location.

    Returns a DataFrame with DatetimeIndex and columns: TG, RR, WS, HU, PP
    """
    end_date   = date.today() - timedelta(days=1)   # yesterday (data available)
    start_date = end_date - timedelta(days=days)

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      ",".join(VARIABLES),
        "timezone":   "UTC",
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "DATE"
    df = df.rename(columns=COL_MAP)
    df = df.interpolate()

    return df


def get_city_name(lat: float, lon: float) -> str:
    """
    Reverse geocode lat/lon to a city name using Open-Meteo geocoding API.
    Falls back to coordinates string if lookup fails.
    """
    try:
        url    = "https://geocoding-api.open-meteo.com/v1/search"
        # Search by nearest location using reverse geocoding via nominatim
        nom_url = "https://nominatim.openstreetmap.org/reverse"
        params  = {
            "lat":    lat,
            "lon":    lon,
            "format": "json",
        }
        headers = {"User-Agent": "MSc-Weather-Forecasting-App/1.0"}
        resp    = requests.get(nom_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data    = resp.json()

        address = data.get("address", {})
        city    = (
            address.get("city") or
            address.get("town") or
            address.get("village") or
            address.get("county") or
            "Unknown Location"
        )
        country = address.get("country", "")
        return f"{city}, {country}" if country else city

    except Exception:
        return f"{lat:.2f}°, {lon:.2f}°"


if __name__ == "__main__":
    # Test with London
    lat, lon = 51.51, -0.13
    print(f"City: {get_city_name(lat, lon)}")
    df = get_recent_weather(lat, lon, days=30)
    print(df.tail())
