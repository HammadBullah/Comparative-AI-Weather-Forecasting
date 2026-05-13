"""
climate_zones.py
----------------
Maps any (latitude, longitude) to one of 5 Köppen climate zones.
Used to select the correct pretrained model for a clicked location.
"""


def get_zone(lat: float, lon: float) -> str:
    """
    Classify a location into one of 5 climate zones based on latitude
    and longitude heuristics.

    Zones: tropical, arid, temperate, continental, polar

    Parameters
    ----------
    lat : float  – latitude  (-90 to 90)
    lon : float  – longitude (-180 to 180)

    Returns
    -------
    str – zone name
    """
    abs_lat = abs(lat)

    # ── Polar ─────────────────────────────────────────────────────────────────
    if abs_lat >= 60:
        return "polar"

    # ── Tropical ──────────────────────────────────────────────────────────────
    if abs_lat <= 15:
        return "tropical"

    # ── Arid (hot desert belt ~15–35°, dry regions) ───────────────────────────
    if 15 < abs_lat <= 35:
        # Middle East, North Africa, Australian outback, SW USA
        arid_lon_bands = [
            (-20 <= lon <= 60),    # North Africa + Middle East
            (110 <= lon <= 155),   # Australia
            (-125 <= lon <= -95),  # SW USA / Mexico
        ]
        if any(arid_lon_bands):
            return "arid"

    # ── Continental (mid-latitude interiors) ──────────────────────────────────
    if 35 < abs_lat < 60:
        continental_lon_bands = [
            (60 <= lon <= 140),    # Central Asia, China, Siberia
            (-110 <= lon <= -70),  # North America interior
            (20 <= lon <= 60),     # Eastern Europe / Russia
        ]
        if any(continental_lon_bands):
            return "continental"

    # ── Temperate (default for mid-latitudes) ─────────────────────────────────
    return "temperate"


# ── Zone metadata ─────────────────────────────────────────────────────────────
ZONE_INFO = {
    "tropical":    {"city": "Singapore",  "color": "#e74c3c"},
    "arid":        {"city": "Dubai",      "color": "#f39c12"},
    "temperate":   {"city": "London",     "color": "#2ecc71"},
    "continental": {"city": "Chicago",    "color": "#3498db"},
    "polar":       {"city": "Reykjavik",  "color": "#9b59b6"},
}


if __name__ == "__main__":
    # Quick sanity check
    tests = [
        (1.3,   103.8,  "tropical"),    # Singapore
        (25.2,   55.3,  "arid"),        # Dubai
        (51.5,   -0.1,  "temperate"),   # London
        (41.9,  -87.6,  "continental"), # Chicago
        (64.1,  -21.9,  "polar"),       # Reykjavik
    ]
    for lat, lon, expected in tests:
        result = get_zone(lat, lon)
        status = "✅" if result == expected else "❌"
        print(f"{status} ({lat}, {lon}) → {result}  (expected {expected})")
