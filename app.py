"""
app.py
------
Global Weather Forecasting Dashboard — MSc Project
Modern dark UI with glassmorphism cards.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import streamlit as st
import folium
from streamlit_folium import st_folium

from climate_zones import get_zone, ZONE_INFO
from live_data     import get_recent_weather, get_city_name
from forecast      import run_all_models

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AtmosCast — Global Weather Forecast",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0a1020 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem; max-width: 1400px; }

/* ── Logo / title bar ── */
.atmoscast-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.2rem;
}
.atmoscast-logo {
    font-size: 2.2rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
}
.atmoscast-logo span {
    color: #4db8ff;
}
.atmoscast-sub {
    font-size: 0.78rem;
    color: #4a6080;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6080;
    margin-bottom: 0.6rem;
}

/* ── City header ── */
.city-name {
    font-size: 2.4rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}
.city-meta {
    font-size: 0.8rem;
    color: #4a6080;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}

/* ── Big temp display ── */
.temp-display {
    font-size: 5rem;
    font-weight: 300;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
}
.temp-unit {
    font-size: 2rem;
    color: #4a6080;
    vertical-align: super;
}

/* ── Stat pill ── */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 30px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #8aa0bc;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stat-pill strong { color: #c8ddf0; }

/* ── Zone badge ── */
.zone-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(77,184,255,0.1);
    border: 1px solid rgba(77,184,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #4db8ff;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Best model badge ── */
.best-badge {
    background: linear-gradient(135deg, rgba(77,184,255,0.15), rgba(77,184,255,0.05));
    border: 1px solid rgba(77,184,255,0.3);
    border-radius: 16px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}
.best-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4db8ff;
    font-weight: 600;
}
.best-model-name {
    font-size: 1.8rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.best-stats {
    font-size: 0.8rem;
    color: #8aa0bc;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}

/* ── Insight box ── */
.insight-box {
    background: rgba(77,184,255,0.05);
    border-left: 3px solid #4db8ff;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.83rem;
    color: #8aa0bc;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* ── 7-day strip ── */
.day-strip {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
}
.day-card {
    flex: 1;
    min-width: 80px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 12px 8px;
    text-align: center;
}
.day-card.best-day {
    background: rgba(77,184,255,0.1);
    border-color: rgba(77,184,255,0.25);
}
.day-name { font-size: 0.68rem; color: #4a6080; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.day-date { font-size: 0.65rem; color: #3a5070; margin-top: 1px; }
.day-temp { font-size: 1.3rem; font-weight: 500; color: #c8ddf0; margin-top: 8px; }
.day-unit { font-size: 0.7rem; color: #4a6080; }

/* ── Model accuracy row ── */
.model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.model-row:last-child { border-bottom: none; }
.model-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}
.model-name { font-size: 0.85rem; font-weight: 500; color: #c8ddf0; }
.model-mae  { font-size: 0.8rem;  font-family: 'DM Mono', monospace; color: #8aa0bc; }
.model-best-tag {
    font-size: 0.65rem; font-weight: 600;
    color: #4db8ff; letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(77,184,255,0.1);
    padding: 2px 8px; border-radius: 10px;
}

/* ── Instructions ── */
.instructions {
    text-align: center;
    padding: 3rem 1rem;
    color: #4a6080;
}
.instructions h2 {
    font-size: 1.1rem; font-weight: 500;
    color: #6a8090; margin-bottom: 0.5rem;
}
.instructions p { font-size: 0.82rem; line-height: 1.7; }

/* ── Streamlit overrides ── */
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #c8ddf0 !important;
}
[data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.72rem !important; }
[data-testid="stDataFrame"]   { border-radius: 12px; overflow: hidden; }
div[data-testid="stSpinner"] > div { color: #4db8ff !important; }

/* Matplotlib chart background to match app */
.stPlotlyChart, [data-testid="stImage"] { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
COLOURS        = {"SARIMA": "#4db8ff", "RF": "#50e3a4", "LSTM": "#f06292"}
VARIABLE       = "TG"
VARIABLE_LABEL = "Temperature (°C)"
UNIT           = "°C"

ZONE_ICONS = {
    "tropical":    "🌴",
    "arid":        "🏜️",
    "temperate":   "🌿",
    "continental": "🌲",
    "polar":       "❄️",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="atmoscast-header">
    <div>
        <div class="atmoscast-logo">Atmos<span>Cast</span></div>
        <div class="atmoscast-sub">Global ML Weather Forecasting · MSc Project 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:1rem 0 1.5rem'></div>",
            unsafe_allow_html=True)

# ── Layout: map | results ──────────────────────────────────────────────────────
col_map, col_results = st.columns([1, 1.6], gap="large")

# ── Map ───────────────────────────────────────────────────────────────────────
with col_map:
    st.markdown('<div class="section-label">📍 Select Location</div>', unsafe_allow_html=True)

    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="CartoDB dark_matter",
    )

    # Style the map
    m.get_root().html.add_child(folium.Element("""
    <style>
    .leaflet-container { border-radius: 20px; }
    </style>
    """))

    map_data = st_folium(
        m,
        height=440,
        width=None,
        returned_objects=["last_clicked"],
    )

    st.markdown(
        "<div style='font-size:0.72rem;color:#3a5070;text-align:center;margin-top:8px'>"
        "Click anywhere on the map to forecast that location</div>",
        unsafe_allow_html=True
    )

# ── Extract click ─────────────────────────────────────────────────────────────
_raw   = map_data.get("last_clicked") if isinstance(map_data, dict) else None
_valid = (
    _raw is not None
    and isinstance(_raw, dict)
    and _raw.get("lat") is not None
    and _raw.get("lng") is not None
)

# ── Results ───────────────────────────────────────────────────────────────────
with col_results:

    if not _valid:
        st.markdown("""
        <div class="glass-card instructions">
            <h2>🌐 Click anywhere on the map</h2>
            <p>
                AtmosCast will fetch real weather data for that location<br>
                and run three machine learning models to forecast<br>
                the next <strong style="color:#c8ddf0">7 days</strong> of temperature.
            </p>
            <br>
            <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:0.5rem">
                <div class="stat-pill">📈 <strong>SARIMA</strong> Seasonal Statistical</div>
                <div class="stat-pill">🌲 <strong>RF</strong> Random Forest</div>
                <div class="stat-pill">🧠 <strong>LSTM</strong> Deep Learning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        lat      = float(_raw["lat"])
        lon      = float(_raw["lng"])
        zone     = get_zone(lat, lon)
        zone_col = ZONE_INFO[zone]["color"]

        # ── City name ─────────────────────────────────────────────────────────
        with st.spinner("Locating..."):
            city = get_city_name(lat, lon)

        # ── Fetch live data ───────────────────────────────────────────────────
        with st.spinner("Fetching weather data..."):
            try:
                live_df = get_recent_weather(lat, lon, days=90)
            except Exception as e:
                st.error(f"Could not fetch weather data: {e}")
                st.stop()

        latest = live_df.iloc[-1]

        # ── City + current conditions card ────────────────────────────────────
        st.markdown(f"""
        <div class="glass-card">
            <div class="zone-badge">
                {ZONE_ICONS.get(zone, "🌍")} {zone.capitalize()} Climate Zone
            </div>
            <div class="city-name">{city}</div>
            <div class="city-meta">{lat:.3f}° N, {lon:.3f}° E · Updated today</div>

            <div style="display:flex;align-items:flex-end;gap:4px;margin:1rem 0 0.2rem">
                <div class="temp-display">{latest['TG']:.0f}</div>
                <div class="temp-unit">{UNIT}</div>
            </div>

            <div class="stat-row">
                <div class="stat-pill">💧 <strong>{latest['HU']:.0f}%</strong> Humidity</div>
                <div class="stat-pill">🌧️ <strong>{latest['RR']:.1f}mm</strong> Rain</div>
                <div class="stat-pill">💨 <strong>{latest['WS']:.0f} km/h</strong> Wind</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Run models ────────────────────────────────────────────────────────
        with st.spinner("Running SARIMA · Random Forest · LSTM..."):
            t0      = time.time()
            results = run_all_models(live_df, zone, variable=VARIABLE)
            elapsed = time.time() - t0

        if not results:
            st.error("Models failed — ensure train_all.py has been run.")
            st.stop()

        best = min(results, key=lambda n: results[n]["mae"])
        b    = results[best]

        # ── Best model card ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="best-badge">
            <div class="best-label">🏆 Best Performing Model</div>
            <div class="best-model-name">{best}</div>
            <div class="best-stats">
                MAE {b['mae']:.3f}{UNIT} &nbsp;·&nbsp;
                RMSE {b['rmse']:.3f}{UNIT} &nbsp;·&nbsp;
                {elapsed:.1f}s total inference
            </div>
        </div>
        """, unsafe_allow_html=True)

        insights = {
            "SARIMA": f"Strong seasonal periodicity detected in this {zone} region. SARIMA's explicit seasonal decomposition outperformed data-driven approaches.",
            "RF":     f"Lag-feature patterns dominate in this {zone} region. Random Forest captured non-linear weather dependencies without needing temporal architecture.",
            "LSTM":   f"Complex long-range dependencies in this {zone} climate. LSTM's sequential memory gave it an edge over statistical and shallow ML models.",
        }
        st.markdown(f'<div class="insight-box">🧠 {insights[best]}</div>', unsafe_allow_html=True)

# ── Below map+results: full width sections ────────────────────────────────────
if _valid and "results" in dir() and results:

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:0.5rem 0 1.5rem'></div>",
                unsafe_allow_html=True)

    left, right = st.columns([1.8, 1], gap="large")

    with left:
        # ── 7-day forecast chart ──────────────────────────────────────────────
        st.markdown('<div class="section-label">📅 7-Day Temperature Forecast</div>',
                    unsafe_allow_html=True)

        series  = live_df[VARIABLE].dropna().resample("D").mean().interpolate()
        context = series.iloc[-10:]

        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor("#0d1528")
        ax.set_facecolor("#0d1528")

        # Actual line
        ax.plot(context.index, context.values,
                color="#ffffff", linewidth=2.2, label="Actual", zorder=5, alpha=0.9)
        ax.fill_between(context.index, context.values,
                        alpha=0.06, color="#ffffff")

        # Forecast lines
        for name, r in results.items():
            cdates = [context.index[-1]] + list(r["future_dates"])
            cvals  = [context.values[-1]] + list(r["future_pred"])
            lw     = 2.5 if name == best else 1.5
            alpha  = 1.0 if name == best else 0.55
            ax.plot(cdates, cvals,
                    color=COLOURS[name], linewidth=lw, linestyle="--",
                    marker="o", markersize=4 if name == best else 3,
                    label=f"{name}  ·  MAE {r['mae']:.2f}{UNIT}",
                    alpha=alpha, zorder=4 if name == best else 3)

        # Forecast shading
        fd_start = context.index[-1]
        fd_end   = list(results.values())[0]["future_dates"][-1]
        ax.axvspan(fd_start, fd_end, alpha=0.04, color="#4db8ff")
        ax.axvline(fd_start, color="#4db8ff", linewidth=0.8, alpha=0.4, linestyle=":")

        # Styling
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.xticks(rotation=0, color="#4a6080", fontsize=8, fontfamily="monospace")
        plt.yticks(color="#4a6080", fontsize=8, fontfamily="monospace")
        ax.set_ylabel("°C", color="#4a6080", fontsize=9)
        ax.tick_params(colors="#4a6080", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#1a2540", linewidth=0.8)
        ax.grid(axis="x", color="#1a2540", linewidth=0.5, alpha=0.5)

        leg = ax.legend(fontsize=8, framealpha=0, labelcolor="#8aa0bc",
                        loc="upper left", handlelength=1.5)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()

        # ── 7-day strip ───────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1rem">📆 Daily Breakdown</div>',
                    unsafe_allow_html=True)

        future_dates = list(results.values())[0]["future_dates"]
        best_preds   = b["future_pred"]

        day_cards_html = '<div class="day-strip">'
        for i, (d, v) in enumerate(zip(future_dates, best_preds)):
            is_best_day = (i == int(np.argmin(np.abs(best_preds - np.mean(best_preds)))))
            card_class  = "day-card best-day" if i == 0 else "day-card"
            day_cards_html += f"""
            <div class="{card_class}">
                <div class="day-name">{d.strftime('%a')}</div>
                <div class="day-date">{d.strftime('%d %b')}</div>
                <div class="day-temp">{v:.0f}<span class="day-unit">{UNIT}</span></div>
            </div>"""
        day_cards_html += "</div>"
        st.markdown(day_cards_html, unsafe_allow_html=True)

    with right:
        # ── Model accuracy ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">📊 Model Comparison</div>',
                    unsafe_allow_html=True)

        model_rows_html = '<div class="glass-card">'
        for name, r in results.items():
            is_best = name == best
            model_rows_html += f"""
            <div class="model-row">
                <div>
                    <span class="model-dot" style="background:{COLOURS[name]}"></span>
                    <span class="model-name">{name}</span>
                    {"<span class='model-best-tag' style='margin-left:8px'>Best</span>" if is_best else ""}
                </div>
                <div class="model-mae">
                    MAE {r['mae']:.3f} · RMSE {r['rmse']:.3f}
                </div>
            </div>"""
        model_rows_html += "</div>"
        st.markdown(model_rows_html, unsafe_allow_html=True)

        # ── MAE bar chart ─────────────────────────────────────────────────────
        names    = list(results.keys())
        mae_vals = [results[n]["mae"] for n in names]
        colours  = [COLOURS[n] for n in names]

        fig2, ax2 = plt.subplots(figsize=(5, 2.6))
        fig2.patch.set_facecolor("#0d1528")
        ax2.set_facecolor("#0d1528")

        bars = ax2.barh(names, mae_vals, color=colours, height=0.45,
                        alpha=0.85, zorder=3)

        # Highlight best
        for bar, name in zip(bars, names):
            if name == best:
                bar.set_alpha(1.0)
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(0.8)

        for bar, val in zip(bars, mae_vals):
            ax2.text(val + max(mae_vals)*0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", color="#8aa0bc",
                     fontsize=8, fontfamily="monospace")

        ax2.set_xlabel("MAE (°C)", color="#4a6080", fontsize=8)
        ax2.tick_params(colors="#8aa0bc", length=0, labelsize=9)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.grid(axis="x", color="#1a2540", linewidth=0.8, zorder=0)
        ax2.set_xlim(0, max(mae_vals) * 1.25)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2)
        plt.close()

        # ── All model forecast table ──────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:0.8rem">📋 All Forecasts</div>',
                    unsafe_allow_html=True)

        fdf = pd.DataFrame({
            "Date": list(results.values())[0]["future_dates"].strftime("%a %d %b")
        })
        for name, r in results.items():
            fdf[f"{name}"] = [f"{v:.1f}°" for v in r["future_pred"]]

        st.dataframe(
            fdf,
            use_container_width=True,
            hide_index=True,
            height=280,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;
            border-top:1px solid rgba(255,255,255,0.05);
            font-size:0.7rem;color:#2a3a50;text-align:center;
            font-family:'DM Mono',monospace;letter-spacing:0.04em">
    ATMOSCAST · MSc Computer Science 2026 · SARIMA · RANDOM FOREST · LSTM · OPEN-METEO
</div>
""", unsafe_allow_html=True)