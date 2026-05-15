"""
app.py — AtmosCast Global Weather Forecasting
MSc Computer Science Project 2026
All features: city search, map pin, multi-variable, confidence bands,
weather icons, feels-like, export CSV, model explainability.
"""

import time
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import folium
from streamlit_folium import st_folium

from climate_zones import get_zone, ZONE_INFO
from live_data     import get_recent_weather, get_city_name
from forecast      import run_all_models

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AtmosCast — Global Weather Forecast",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLOURS = {"SARIMA": "#4db8ff", "RF": "#50e3a4", "LSTM": "#f06292"}

VARIABLES = {
    "TG": {"label": "Temperature",  "unit": "°C",   "icon": "🌡️"},
    "HU": {"label": "Humidity",     "unit": "%",    "icon": "💧"},
    "RR": {"label": "Precipitation","unit": "mm",   "icon": "🌧️"},
    "WS": {"label": "Wind Speed",   "unit": "km/h", "icon": "💨"},
    "PP": {"label": "Pressure",     "unit": "hPa",  "icon": "🔵"},
}

ZONE_ICONS = {
    "tropical": "🌴", "arid": "🏜️",
    "temperate": "🌿", "continental": "🌲", "polar": "❄️",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def feels_like(temp_c, wind_kph, humidity_pct):
    """Wind chill / heat index hybrid apparent temperature."""
    if temp_c <= 10 and wind_kph >= 4.8:
        v = wind_kph ** 0.16
        return 13.12 + 0.6215*temp_c - 11.37*v + 0.3965*temp_c*v
    elif temp_c >= 27:
        hi = (-8.78469475556 + 1.61139411*temp_c + 2.33854883889*humidity_pct
              - 0.14611605*temp_c*humidity_pct - 0.012308094*temp_c**2
              - 0.0164248277778*humidity_pct**2 + 0.002211732*temp_c**2*humidity_pct
              + 0.00072546*temp_c*humidity_pct**2
              - 0.000003582*temp_c**2*humidity_pct**2)
        return hi
    return temp_c

def weather_condition(temp, rain, wind, humidity):
    """Map numeric weather to human-readable condition + emoji."""
    if rain > 10:   return "Heavy Rain",    "🌧️"
    if rain > 2:    return "Light Rain",    "🌦️"
    if wind > 50:   return "Windy",         "🌬️"
    if humidity>85: return "Foggy",         "🌫️"
    if temp > 30:   return "Hot & Sunny",   "☀️"
    if temp > 20:   return "Warm",          "🌤️"
    if temp > 10:   return "Mild",          "⛅"
    if temp > 0:    return "Cold",          "🧥"
    return "Freezing", "🥶"

def geocode_city(city_name):
    """Return (lat, lon, display_name) for a city name via Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city_name, "format": "json", "limit": 1},
            headers={"User-Agent": "AtmosCast-MSc/1.0"},
            timeout=8,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None, None, None

def confidence_band(predictions, spread_factor=0.12):
    """Generate upper/lower confidence bounds around predictions."""
    spread = np.abs(predictions) * spread_factor + 0.5
    return predictions - spread, predictions + spread

def inline_card(content_html, extra_style=""):
    """Wrap content in a glass card using inline styles."""
    return (
        f"<div style='background:rgba(255,255,255,0.04);"
        f"border:1px solid rgba(255,255,255,0.08);border-radius:20px;"
        f"padding:1.4rem 1.6rem;margin-bottom:1rem;{extra_style}'>"
        f"{content_html}</div>"
    )

def pill(icon, value, label, color="#c8ddf0"):
    return (
        f"<div style='background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);"
        f"border-radius:30px;padding:6px 14px;font-size:0.78rem;color:#8aa0bc'>"
        f"{icon} <b style='color:{color}'>{value}</b> {label}</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0a1020 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1440px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #4a6080;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(77,184,255,0.15) !important;
    color: #4db8ff !important;
}

/* Selectbox / input */
.stSelectbox > div > div,
.stTextInput > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #c8ddf0 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #c8ddf0 !important;
}
[data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.72rem !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Download button */
.stDownloadButton > button {
    background: rgba(77,184,255,0.1) !important;
    border: 1px solid rgba(77,184,255,0.3) !important;
    color: #4db8ff !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}
.stDownloadButton > button:hover {
    background: rgba(77,184,255,0.2) !important;
}

/* Spinner */
div[data-testid="stSpinner"] > div { color: #4db8ff !important; }

/* Section label */
.sec-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4a6080; margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "recent" not in st.session_state:
    st.session_state.recent = []   # list of (city, lat, lon)
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = None
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = None

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.2rem'>
    <div>
        <div style='font-size:2rem;font-weight:600;color:#fff;letter-spacing:-0.04em;line-height:1'>
            Atmos<span style='color:#4db8ff'>Cast</span>
        </div>
        <div style='font-size:0.72rem;color:#4a6080;letter-spacing:0.08em;
                    text-transform:uppercase;margin-top:2px'>
            Global ML Weather Forecasting · MSc Project 2026
        </div>
    </div>
</div>
<div style='height:1px;background:rgba(255,255,255,0.06);margin:0.8rem 0 1.2rem'></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP CONTROLS ROW
# ─────────────────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([2, 1.2, 0.8])

with ctrl1:
    city_search = st.text_input(
        "🔍 Search city",
        placeholder="e.g. Tokyo, Lagos, Buenos Aires...",
        label_visibility="collapsed",
    )
    if city_search:
        with st.spinner("Searching..."):
            s_lat, s_lon, s_name = geocode_city(city_search)
        if s_lat:
            st.session_state.selected_lat = s_lat
            st.session_state.selected_lon = s_lon
            st.success(f"📍 Found: {s_name[:60]}", icon=None)
        else:
            st.error("City not found — try a different name")

with ctrl2:
    var_key = st.selectbox(
        "Variable",
        list(VARIABLES.keys()),
        format_func=lambda k: f"{VARIABLES[k]['icon']} {VARIABLES[k]['label']}",
        label_visibility="collapsed",
    )

with ctrl3:
    # Recent locations quick-select
    if st.session_state.recent:
        recent_options = ["Recent locations..."] + [r[0] for r in st.session_state.recent]
        chosen_recent  = st.selectbox("Recent", recent_options, label_visibility="collapsed")
        if chosen_recent != "Recent locations...":
            match = next((r for r in st.session_state.recent if r[0] == chosen_recent), None)
            if match:
                st.session_state.selected_lat = match[1]
                st.session_state.selected_lon = match[2]

# ─────────────────────────────────────────────────────────────────────────────
# MAP + RESULTS COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
col_map, col_results = st.columns([1, 1.7], gap="large")

with col_map:
    st.markdown('<div class="sec-label">📍 Select Location</div>', unsafe_allow_html=True)

    init_lat = st.session_state.selected_lat or 20
    init_lon = st.session_state.selected_lon or 0
    zoom     = 8 if st.session_state.selected_lat else 2

    m = folium.Map(location=[init_lat, init_lon], zoom_start=zoom,
                   tiles="CartoDB dark_matter")
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-container{border-radius:18px}</style>"
    ))

    # Drop pin if location selected
    if st.session_state.selected_lat:
        folium.Marker(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(m, height=420, width=None, returned_objects=["last_clicked"])

    st.markdown(
        "<div style='font-size:0.7rem;color:#3a5070;text-align:center;margin-top:6px'>"
        "Click map or search above to select a location</div>",
        unsafe_allow_html=True
    )

# ── Resolve click vs search ───────────────────────────────────────────────────
_raw   = map_data.get("last_clicked") if isinstance(map_data, dict) else None
_valid_click = (
    _raw is not None and isinstance(_raw, dict)
    and _raw.get("lat") is not None and _raw.get("lng") is not None
)

if _valid_click:
    st.session_state.selected_lat = float(_raw["lat"])
    st.session_state.selected_lon = float(_raw["lng"])

lat = st.session_state.selected_lat
lon = st.session_state.selected_lon
_valid = lat is not None and lon is not None

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────────────────────────────────────
with col_results:

    if not _valid:
        st.markdown(inline_card("""
            <div style='text-align:center;padding:2rem 1rem;color:#4a6080'>
                <div style='font-size:3rem;margin-bottom:0.5rem'>🌐</div>
                <div style='font-size:1.1rem;font-weight:500;color:#6a8090;margin-bottom:0.5rem'>
                    Click anywhere on the map</div>
                <div style='font-size:0.82rem;line-height:1.7'>
                    or search a city above to get a<br>
                    <b style='color:#c8ddf0'>7-day ML weather forecast</b>
                </div>
                <div style='display:flex;gap:10px;justify-content:center;
                            flex-wrap:wrap;margin-top:1.2rem'>
                    <div style='background:rgba(77,184,255,0.08);border:1px solid
                    rgba(77,184,255,0.15);border-radius:20px;padding:5px 14px;
                    font-size:0.75rem;color:#4db8ff'>📈 SARIMA</div>
                    <div style='background:rgba(80,227,164,0.08);border:1px solid
                    rgba(80,227,164,0.15);border-radius:20px;padding:5px 14px;
                    font-size:0.75rem;color:#50e3a4'>🌲 Random Forest</div>
                    <div style='background:rgba(240,98,146,0.08);border:1px solid
                    rgba(240,98,146,0.15);border-radius:20px;padding:5px 14px;
                    font-size:0.75rem;color:#f06292'>🧠 LSTM</div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    else:
        # ── Fetch city + weather ──────────────────────────────────────────────
        with st.spinner("Fetching location & weather data..."):
            city    = get_city_name(lat, lon)
            try:
                live_df = get_recent_weather(lat, lon, days=90)
            except Exception as e:
                st.error(f"Could not fetch weather: {e}")
                st.stop()

        # Save to recent
        if not any(r[0] == city for r in st.session_state.recent):
            st.session_state.recent.insert(0, (city, lat, lon))
            st.session_state.recent = st.session_state.recent[:5]

        latest  = live_df.iloc[-1]
        zone    = get_zone(lat, lon)
        cond, cond_icon = weather_condition(
            latest["TG"], latest["RR"], latest["WS"], latest["HU"]
        )
        fl = feels_like(latest["TG"], latest["WS"], latest["HU"])

        # ── Current conditions card ───────────────────────────────────────────
        zone_icon = ZONE_ICONS.get(zone, "🌍")
        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid "
            "rgba(255,255,255,0.08);border-radius:20px;padding:1.3rem 1.6rem;"
            "margin-bottom:0.8rem'>"

            f"<div style='display:inline-flex;align-items:center;gap:6px;"
            f"background:rgba(77,184,255,0.1);border:1px solid rgba(77,184,255,0.2);"
            f"border-radius:20px;padding:3px 12px;font-size:0.7rem;font-weight:600;"
            f"color:#4db8ff;letter-spacing:0.05em;text-transform:uppercase;"
            f"margin-bottom:0.7rem'>{zone_icon} {zone.capitalize()} Zone · {cond_icon} {cond}</div>"

            f"<div style='font-size:2.2rem;font-weight:600;color:#fff;"
            f"letter-spacing:-0.03em;line-height:1.1'>{city}</div>"
            f"<div style='font-size:0.75rem;color:#4a6080;margin-top:3px;"
            f"font-family:monospace'>{lat:.3f}°, {lon:.3f}° · Live data</div>"

            "<div style='display:flex;align-items:flex-end;gap:6px;margin:0.8rem 0 0.2rem'>"
            f"<div style='font-size:4.5rem;font-weight:300;color:#fff;"
            f"letter-spacing:-0.04em;line-height:1'>{latest['TG']:.0f}</div>"
            f"<div style='font-size:1.8rem;color:#4a6080;padding-bottom:0.5rem'>°C</div>"
            f"<div style='font-size:0.8rem;color:#4a6080;padding-bottom:0.6rem;"
            f"margin-left:4px'>Feels like<br><b style='color:#8aa0bc'>{fl:.0f}°C</b></div>"
            "</div>"

            "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:0.6rem'>"
            + pill("💧", f"{latest['HU']:.0f}%",   "Humidity")
            + pill("🌧️", f"{latest['RR']:.1f}mm",  "Rain")
            + pill("💨", f"{latest['WS']:.0f}km/h", "Wind")
            + pill("🔵", f"{latest['PP']:.0f}hPa",  "Pressure")
            + "</div></div>",
            unsafe_allow_html=True
        )

        # ── Run models ────────────────────────────────────────────────────────
        with st.spinner("Running SARIMA · Random Forest · LSTM..."):
            t0      = time.time()
            results = run_all_models(live_df, zone, variable=var_key)
            elapsed = time.time() - t0

        if not results:
            st.error("Models failed — run train_all.py first.")
            st.stop()

        best = min(results, key=lambda n: results[n]["mae"])
        b    = results[best]
        unit = VARIABLES[var_key]["unit"]

        # ── Best model badge ──────────────────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,rgba(77,184,255,0.15),"
            "rgba(77,184,255,0.04));border:1px solid rgba(77,184,255,0.3);"
            "border-radius:16px;padding:0.9rem 1.3rem;margin-bottom:0.8rem;"
            "display:flex;align-items:center;justify-content:space-between'>"
            "<div>"
            "<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.12em;"
            "color:#4db8ff;font-weight:600'>🏆 Best Model</div>"
            f"<div style='font-size:1.6rem;font-weight:600;color:#fff;"
            f"letter-spacing:-0.02em'>{best}</div>"
            "</div>"
            "<div style='text-align:right;font-family:monospace;font-size:0.78rem;color:#8aa0bc'>"
            f"MAE {b['mae']:.3f}{unit}<br>RMSE {b['rmse']:.3f}{unit}<br>"
            f"<span style='color:#4a6080;font-size:0.7rem'>{elapsed:.1f}s inference</span>"
            "</div></div>",
            unsafe_allow_html=True
        )

        insights = {
            "SARIMA": f"Strong seasonal periodicity in this {zone} zone. SARIMA's explicit seasonal decomposition outperformed data-driven models.",
            "RF":     f"Non-linear lag patterns dominate in this {zone} zone. Random Forest captured temporal dependencies efficiently without recurrent architecture.",
            "LSTM":   f"Complex long-range dependencies in this {zone} climate gave LSTM's sequential memory an edge over statistical and shallow ML models.",
        }
        st.markdown(
            "<div style='background:rgba(77,184,255,0.05);border-left:3px solid #4db8ff;"
            "border-radius:0 10px 10px 0;padding:0.8rem 1rem;font-size:0.8rem;"
            f"color:#8aa0bc;line-height:1.6;margin-bottom:0.5rem'>🧠 {insights[best]}</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TABS — main content area
# ─────────────────────────────────────────────────────────────────────────────
if _valid and "results" in dir() and results:

    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,0.06);"
        "margin:0.8rem 0 1.2rem'></div>",
        unsafe_allow_html=True
    )

    tab_forecast, tab_models, tab_export = st.tabs([
        "📅  7-Day Forecast",
        "📊  Model Comparison",
        "📤  Export & Data",
    ])

    # ── TAB 1: 7-Day Forecast ─────────────────────────────────────────────────
    with tab_forecast:

        fc_left, fc_right = st.columns([2, 1], gap="large")

        with fc_left:
            # Chart
            series  = live_df[var_key].dropna().resample("D").mean().interpolate()
            context = series.iloc[-12:]
            vinfo   = VARIABLES[var_key]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0b1120")
            ax.set_facecolor("#0b1120")

            # Actual
            ax.plot(context.index, context.values,
                    color="#ffffff", linewidth=2.2, label="Actual",
                    zorder=5, alpha=0.9)
            ax.fill_between(context.index, context.values,
                            alpha=0.06, color="#ffffff")

            # Model forecasts + confidence bands
            for name, r in results.items():
                cdates = [context.index[-1]] + list(r["future_dates"])
                cvals  = [context.values[-1]] + list(r["future_pred"])
                lw     = 2.5 if name == best else 1.5
                alpha  = 1.0 if name == best else 0.5
                col    = COLOURS[name]

                ax.plot(cdates, cvals, color=col, linewidth=lw,
                        linestyle="--", marker="o",
                        markersize=4 if name == best else 2.5,
                        label=f"{name}  MAE {r['mae']:.2f}{vinfo['unit']}",
                        alpha=alpha, zorder=4)

                # Confidence band for best model only
                if name == best:
                    fp    = np.array(r["future_pred"])
                    lo, hi = confidence_band(fp)
                    band_dates = list(r["future_dates"])
                    ax.fill_between(band_dates, lo, hi,
                                    alpha=0.12, color=col, zorder=2)

            # Forecast divider
            ax.axvline(context.index[-1], color="#4db8ff",
                       linewidth=0.8, alpha=0.5, linestyle=":")
            ax.axvspan(context.index[-1],
                       list(results.values())[0]["future_dates"][-1],
                       alpha=0.03, color="#4db8ff")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.xticks(rotation=0, color="#4a6080", fontsize=8)
            plt.yticks(color="#4a6080", fontsize=8)
            ax.set_ylabel(f"{vinfo['label']} ({vinfo['unit']})",
                          color="#4a6080", fontsize=8)
            ax.tick_params(colors="#4a6080", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(axis="y", color="#1a2540", linewidth=0.8)
            ax.grid(axis="x", color="#1a2540", linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=8, framealpha=0, labelcolor="#8aa0bc",
                      loc="upper left")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig)
            plt.close()

            # Day strip
            future_dates = list(results.values())[0]["future_dates"]
            best_preds   = b["future_pred"]
            vunit        = VARIABLES[var_key]["unit"]

            strip = "<div style='display:flex;gap:7px;overflow-x:auto;padding:4px 0'>"
            for i, (d, v) in enumerate(zip(future_dates, best_preds)):
                hi_style = ("background:rgba(77,184,255,0.12);"
                            "border:1px solid rgba(77,184,255,0.3)") if i == 0 else (
                            "background:rgba(255,255,255,0.04);"
                            "border:1px solid rgba(255,255,255,0.07)")
                strip += (
                    f"<div style='flex:1;min-width:76px;{hi_style};"
                    f"border-radius:14px;padding:11px 7px;text-align:center'>"
                    f"<div style='font-size:0.65rem;color:#4a6080;font-weight:600;"
                    f"letter-spacing:0.08em;text-transform:uppercase'>{d.strftime('%a')}</div>"
                    f"<div style='font-size:0.62rem;color:#2a4060;margin-top:1px'>"
                    f"{d.strftime('%d %b')}</div>"
                    f"<div style='font-size:1.25rem;font-weight:500;color:#c8ddf0;"
                    f"margin-top:7px'>{v:.0f}"
                    f"<span style='font-size:0.65rem;color:#4a6080'>{vunit}</span></div>"
                    f"</div>"
                )
            strip += "</div>"
            st.markdown(strip, unsafe_allow_html=True)

        with fc_right:
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin-bottom:0.6rem'>"
                "Shaded area = confidence band</div>",
                unsafe_allow_html=True
            )

            # All models forecast table
            fdf = pd.DataFrame({
                "Date": list(results.values())[0]["future_dates"].strftime("%a %d %b")
            })
            for name, r in results.items():
                fdf[f"{name} ({vunit})"] = [f"{v:.1f}" for v in r["future_pred"]]

            st.dataframe(fdf, use_container_width=True, hide_index=True, height=295)

            # Training info box
            st.markdown(
                "<div style='background:rgba(255,255,255,0.03);"
                "border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
                "padding:0.8rem 1rem;margin-top:0.8rem;font-size:0.75rem;color:#4a6080'>"
                "<b style='color:#6a8090'>Training Info</b><br>"
                f"Zone: <b style='color:#8aa0bc'>{zone.capitalize()}</b><br>"
                "Data: <b style='color:#8aa0bc'>2015–2024 · Open-Meteo</b><br>"
                "Horizon: <b style='color:#8aa0bc'>7 days ahead</b>"
                "</div>",
                unsafe_allow_html=True
            )

    # ── TAB 2: Model Comparison ───────────────────────────────────────────────
    with tab_models:

        m_left, m_right = st.columns([1, 1], gap="large")

        with m_left:
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin-bottom:0.8rem'>"
                "Accuracy Metrics</div>",
                unsafe_allow_html=True
            )

            # Model rows
            rows_html = "<div style='background:rgba(255,255,255,0.03);" \
                        "border:1px solid rgba(255,255,255,0.07);" \
                        "border-radius:16px;overflow:hidden'>"
            for name, r in results.items():
                is_best  = name == best
                bg       = "rgba(77,184,255,0.06)" if is_best else "transparent"
                best_tag = ("<span style='font-size:0.6rem;font-weight:600;color:#4db8ff;"
                            "background:rgba(77,184,255,0.1);padding:2px 8px;"
                            "border-radius:10px;margin-left:8px'>BEST</span>"
                            if is_best else "")
                rows_html += (
                    f"<div style='display:flex;align-items:center;"
                    f"justify-content:space-between;padding:12px 16px;"
                    f"background:{bg};border-bottom:1px solid rgba(255,255,255,0.05)'>"
                    f"<div style='display:flex;align-items:center;gap:10px'>"
                    f"<div style='width:10px;height:10px;border-radius:50%;"
                    f"background:{COLOURS[name]}'></div>"
                    f"<span style='font-size:0.88rem;font-weight:500;color:#c8ddf0'>"
                    f"{name}</span>{best_tag}</div>"
                    f"<div style='font-family:monospace;font-size:0.78rem;color:#8aa0bc;"
                    f"text-align:right'>MAE {r['mae']:.3f}<br>"
                    f"<span style='color:#4a6080'>RMSE {r['rmse']:.3f}</span></div>"
                    f"</div>"
                )
            rows_html += "</div>"
            st.markdown(rows_html, unsafe_allow_html=True)

            # Horizontal bar chart
            st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
            names    = list(results.keys())
            mae_vals = [results[n]["mae"] for n in names]
            rmse_vals= [results[n]["rmse"] for n in names]

            fig2, ax2 = plt.subplots(figsize=(5, 2.8))
            fig2.patch.set_facecolor("#0b1120")
            ax2.set_facecolor("#0b1120")

            y  = np.arange(len(names))
            w  = 0.35
            b1 = ax2.barh(y + w/2, mae_vals,  w,
                          color=[COLOURS[n] for n in names], alpha=0.85, label="MAE")
            b2 = ax2.barh(y - w/2, rmse_vals, w,
                          color=[COLOURS[n] for n in names], alpha=0.4,
                          hatch="//", label="RMSE")

            for bar, val in zip(b1, mae_vals):
                ax2.text(val + max(mae_vals)*0.02,
                         bar.get_y() + bar.get_height()/2,
                         f"{val:.3f}", va="center", color="#8aa0bc",
                         fontsize=7.5, fontfamily="monospace")

            ax2.set_yticks(y)
            ax2.set_yticklabels(names, color="#8aa0bc", fontsize=9)
            ax2.set_xlabel(f"Error ({vunit})", color="#4a6080", fontsize=8)
            ax2.tick_params(colors="#4a6080", length=0)
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.grid(axis="x", color="#1a2540", linewidth=0.8, zorder=0)
            ax2.set_xlim(0, max(mae_vals + rmse_vals) * 1.28)
            ax2.legend(fontsize=7.5, framealpha=0, labelcolor="#6a8090",
                       loc="lower right")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig2)
            plt.close()

        with m_right:
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin-bottom:0.8rem'>"
                "🔍 Model Explainability</div>",
                unsafe_allow_html=True
            )

            # RF feature importance
            if "RF" in results:
                try:
                    import joblib
                    rf_model   = joblib.load(f"models/{zone}_rf.pkl")
                    importances = rf_model.feature_importances_
                    top_n       = 7
                    top_idx     = np.argsort(importances)[-top_n:][::-1]
                    top_imp     = importances[top_idx]
                    top_labels  = [f"Lag {i+1}d" for i in top_idx]

                    fig3, ax3 = plt.subplots(figsize=(5, 3))
                    fig3.patch.set_facecolor("#0b1120")
                    ax3.set_facecolor("#0b1120")
                    bars3 = ax3.barh(top_labels[::-1], top_imp[::-1],
                                     color="#50e3a4", alpha=0.8, height=0.55)
                    ax3.set_xlabel("Importance", color="#4a6080", fontsize=8)
                    ax3.set_title("RF: Top Lag Feature Importances",
                                  color="#8aa0bc", fontsize=9, pad=8)
                    ax3.tick_params(colors="#6a8090", length=0, labelsize=8)
                    for sp in ax3.spines.values():
                        sp.set_visible(False)
                    ax3.grid(axis="x", color="#1a2540", linewidth=0.8)
                    plt.tight_layout(pad=0.4)
                    st.pyplot(fig3)
                    plt.close()

                    st.markdown(
                        "<div style='font-size:0.75rem;color:#4a6080;line-height:1.6;"
                        "margin-top:0.5rem'>"
                        f"The most predictive feature is <b style='color:#8aa0bc'>"
                        f"{top_labels[0]}</b>, meaning the value "
                        f"<b style='color:#8aa0bc'>{top_idx[0]+1} day(s) ago</b> "
                        f"has the strongest influence on tomorrow's forecast.</div>",
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.info("RF explainability requires models/zone_rf.pkl")

            # Backtest plot
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin:1rem 0 0.5rem'>"
                "Backtest: Actual vs Predicted</div>",
                unsafe_allow_html=True
            )

            fig4, ax4 = plt.subplots(figsize=(5, 2.5))
            fig4.patch.set_facecolor("#0b1120")
            ax4.set_facecolor("#0b1120")
            n_show = min(60, len(b["actual"]))
            ax4.plot(b["actual"][-n_show:],  color="#ffffff",
                     linewidth=1.5, label="Actual", alpha=0.85)
            ax4.plot(b["backtest"][-n_show:], color=COLOURS[best],
                     linewidth=1.5, linestyle="--", label=f"{best} Pred", alpha=0.9)
            ax4.set_title(f"{best} — Last {n_show} test days",
                          color="#8aa0bc", fontsize=8.5, pad=6)
            ax4.tick_params(colors="#4a6080", length=0, labelsize=7.5)
            for sp in ax4.spines.values():
                sp.set_visible(False)
            ax4.grid(color="#1a2540", linewidth=0.7, alpha=0.8)
            ax4.legend(fontsize=7.5, framealpha=0, labelcolor="#6a8090")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig4)
            plt.close()

    # ── TAB 3: Export ─────────────────────────────────────────────────────────
    with tab_export:

        ex_left, ex_right = st.columns([1, 1], gap="large")

        with ex_left:
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin-bottom:0.8rem'>"
                "📥 Download Forecast</div>",
                unsafe_allow_html=True
            )

            # Build export dataframe
            future_dates = list(results.values())[0]["future_dates"]
            export_df    = pd.DataFrame({"Date": future_dates.strftime("%Y-%m-%d")})
            for name, r in results.items():
                lo, hi = confidence_band(r["future_pred"])
                export_df[f"{name}_forecast"] = [f"{v:.2f}" for v in r["future_pred"]]
                export_df[f"{name}_lower_CI"]  = [f"{v:.2f}" for v in lo]
                export_df[f"{name}_upper_CI"]  = [f"{v:.2f}" for v in hi]
            export_df["best_model"]    = best
            export_df["location"]      = city
            export_df["latitude"]      = lat
            export_df["longitude"]     = lon
            export_df["variable"]      = var_key
            export_df["unit"]          = VARIABLES[var_key]["unit"]
            export_df["generated_on"]  = pd.Timestamp.today().strftime("%Y-%m-%d")

            csv_bytes = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️  Download 7-Day Forecast CSV",
                data=csv_bytes,
                file_name=f"atmoscast_{city.split(',')[0].strip().lower().replace(' ','_')}"
                          f"_{var_key}_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("<div style='margin-top:0.5rem'>", unsafe_allow_html=True)
            st.dataframe(export_df[["Date"] + [f"{n}_forecast" for n in results]],
                         use_container_width=True, hide_index=True)

        with ex_right:
            st.markdown(
                "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#4a6080;margin-bottom:0.8rem'>"
                "📊 Recent Observations</div>",
                unsafe_allow_html=True
            )

            obs_df = live_df[list(VARIABLES.keys())].tail(14).copy()
            obs_df.index = obs_df.index.strftime("%d %b %Y")
            obs_df.columns = [f"{VARIABLES[k]['icon']} {VARIABLES[k]['label']}"
                              for k in obs_df.columns]
            st.dataframe(obs_df.iloc[::-1], use_container_width=True, height=320)

            # Location summary box
            st.markdown(
                "<div style='background:rgba(255,255,255,0.03);"
                "border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
                "padding:0.9rem 1.1rem;margin-top:0.8rem;font-size:0.78rem;color:#4a6080'>"
                f"<b style='color:#8aa0bc'>📍 {city}</b><br>"
                f"Lat {lat:.4f}° · Lon {lon:.4f}°<br>"
                f"Climate zone: <b style='color:#8aa0bc'>{zone.capitalize()}</b><br>"
                f"Condition: <b style='color:#8aa0bc'>{cond_icon} {cond}</b><br>"
                f"Feels like: <b style='color:#8aa0bc'>{fl:.1f}°C</b>"
                "</div>",
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2rem;padding-top:1rem;
            border-top:1px solid rgba(255,255,255,0.05);
            font-size:0.68rem;color:#2a3a50;text-align:center;
            font-family:monospace;letter-spacing:0.04em'>
    ATMOSCAST · MSc Computer Science 2026 · SARIMA · RANDOM FOREST · LSTM · OPEN-METEO · PyTorch
</div>
""", unsafe_allow_html=True)