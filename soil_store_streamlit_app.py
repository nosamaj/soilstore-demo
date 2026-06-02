import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ICM Soil Store Explorer", layout="wide")

st.title("ICM Soil Store Explorer (Soil Store Only)")
st.caption("Interactive soil-store response explorer using synthetic storms (1 ha catchment) indicative only")

# ======================
# SIDEBAR CONTROLS
# ======================

with st.sidebar:
    st.header("Synthetic storm")

    storm_type = st.selectbox("Storm shape", ["Triangular", "Block", "Double peak"])
    duration_hr = st.slider("Storm duration (hr)", 1.0, 48.0, 12.0)
    peak_intensity_mmhr = st.slider("Peak intensity (mm/hr)", 1.0, 100.0, 20.0)
    start_hr = st.slider("Storm start time (hr)", 0.0, 24.0, 2.0)
    total_sim_hr = st.slider("Simulation length (hr)", 6.0, 168.0, 48.0)
    dt_min = st.slider("Time step (min)", 1, 60, 5)

    st.markdown("---")

    st.header("Surface routing (simplified)")

    runoff_fraction = st.slider("Runoff fraction (PR)", 0.0, 1.0, 0.2)
    contributing_area_pct = st.slider(
        "Proportion of area contributing to soil store (%)",
        0.0, 100.0, 50.0
    )
    depression_storage_mm = st.slider("Depression storage (mm)", 0.0, 20.0, 2.0)

    st.markdown("---")

    st.header("Soil store parameters")

    soil_depth_m = st.slider("Soil depth (m)", 0.1, 5.0, 1.0)
    soil_porosity_pct = st.slider("Soil porosity (%)", 1.0, 100.0, 50.0)
    percolation_threshold_pct = st.slider("Percolation threshold (%)", 0.0, 100.0, 50.0)
    percolation_coefficient = st.slider("Percolation coefficient (1/day)", 0.1, 10.0, 1.0)
    percolation_percentage = st.slider("Percolation % infiltrating", 0.0, 100.0, 5.0)
    initial_saturation = st.slider("Initial soil saturation (%)", 0.0, 100.0, 10.0)

    potential_evap_mmday = st.slider("Potential evaporation (mm/day)", 0.0, 10.0, 1.5)

# ======================
# STORM GENERATOR
# ======================

def make_storm(t, storm_type, start, duration, peak):
    rain = np.zeros_like(t)
    tt = t - start
    active = (tt >= 0) & (tt <= duration)

    if storm_type == "Block":
        rain[active] = peak

    elif storm_type == "Triangular":
        mid = duration / 2
        left = active & (tt <= mid)
        right = active & (tt > mid)

        rain[left] = peak * tt[left] / mid
        rain[right] = peak * (1 - (tt[right] - mid) / mid)

    else:  # double peak
        f = duration * 0.35
        gap = duration * 0.3

        peak1 = (tt >= 0) & (tt <= f)
        peak2 = (tt >= f + gap) & (tt <= f + gap + f)

        rain[peak1] = peak * (tt[peak1] / f)
        rain[peak2] = peak * ((tt[peak2] - (f + gap)) / f)

    return rain

# ======================
# MODEL
# ======================

def run_model():
    A = 10000.0  # 1 hectare

    dt_s = dt_min * 60
    dt_hr = dt_min / 60

    t = np.arange(0, total_sim_hr, dt_hr)
    rain_mmhr = make_storm(t, storm_type, start_hr, duration_hr, peak_intensity_mmhr)

    n = len(t)

    # states
    D = np.zeros(n)

    # flows
    q_perc = np.zeros(n)
    q_ri = np.zeros(n)
    q_soil = np.zeros(n)

    # params
    Psoil = soil_porosity_pct / 100
    alpha = percolation_percentage / 100
    area_factor = contributing_area_pct / 100

    Dmax = soil_depth_m
    Dt = (percolation_threshold_pct / 100) * Dmax

    
    tau = percolation_coefficient * 86400  # seconds
    k_eff = 1.0 / tau
    # convert to 1/s
    evap_full = potential_evap_mmday / 1000 / 86400

    D[0] = (initial_saturation / 100) * Dmax

    dep_store = 0

    for i in range(1, n):
        rain_mps = rain_mmhr[i] / 1000 / 3600
        rainfall_depth = rain_mps * dt_s

        # depression storage
        available = max(depression_storage_mm / 1000 - dep_store, 0)
        fill = min(rainfall_depth, available)
        dep_store += fill

        effective = rainfall_depth - fill

        # === KEY NEW LOGIC ===
        soil_depth = (1 - runoff_fraction) * area_factor * effective
        runoff_depth = (
            runoff_fraction * effective +
            (1 - area_factor) * effective
        )

        q_soil[i] = soil_depth * A / dt_s

        # evaporation
        sat = D[i-1] / Dmax if Dmax > 0 else 0
        q_evap = evap_full * sat * A

        # percolation
        if D[i-1] < Dt:
            q_perc[i] = 0
        else:
            q_perc[i] = k_eff * A * (D[i-1] - Dt)

        q_ri[i] = alpha * q_perc[i] * Psoil

        dDdt = (q_soil[i] - q_evap - q_perc[i]) / (Psoil * A)

        D[i] = np.clip(D[i-1] + dDdt * dt_s, 0, Dmax)

    df = pd.DataFrame({
        "time_hr": t,
        "rain": rain_mmhr,
        "soil_depth": D,
        "saturation_pct": 100 * D / Dmax,
        "percolation_lps": q_perc * 1000,
        "ri_infiltration_lps": q_ri * 1000,
        "soil_inflow_lps": q_soil * 1000
    })

    return df, Dt

df, Dt = run_model()

# ======================
# PLOTS
# ======================

st.subheader("Storm")
fig = go.Figure()
fig.add_bar(x=df.time_hr, y=df.rain)
fig.update_layout(yaxis_title="mm/hr")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Soil Store")
fig = go.Figure()
fig.add_scatter(x=df.time_hr, y=df.soil_depth, name="Soil depth")
fig.add_scatter(x=df.time_hr, y=[Dt]*len(df), name="Threshold", line=dict(dash="dash"))
fig.update_layout(yaxis_title="Depth (m)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Flows (1 ha)")
fig = go.Figure()
fig.add_scatter(x=df.time_hr, y=df.percolation_lps, name="Percolation")
fig.add_scatter(x=df.time_hr, y=df.ri_infiltration_lps, name="RI infiltration")
fig.add_scatter(x=df.time_hr, y=df.soil_inflow_lps, name="Soil inflow")
fig.update_layout(yaxis_title="Flow (L/s over 1 ha)")
st.plotly_chart(fig, use_container_width=True)

# ======================
# METRICS
# ======================

st.subheader("Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Peak depth (m)", f"{df.soil_depth.max():.2f}")
col1.metric("Peak saturation (%)", f"{df.saturation_pct.max():.1f}")

col2.metric("Peak percolation (L/s)", f"{df.percolation_lps.max():.1f}")
col2.metric("Peak RI infiltration (L/s)", f"{df.ri_infiltration_lps.max():.1f}")

col3.metric("Time above threshold (hr)",
            f"{(df.soil_depth >= Dt).sum() * (dt_min/60):.2f}")

col3.metric("Effective contributing area (ha)",
            f"{contributing_area_pct/100:.2f}")

# ======================
# DOWNLOAD
# ======================

csv = df.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "soil_store.csv")

# ======================
# NOTES
# ======================

with st.expander("Model notes"):
    st.markdown("""
- Catchment area = **1 hectare**
- Soil store implemented as thresholded linear reservoir
- Percolation only occurs above threshold
- Simplified rainfall partitioning used for clarity
- Intended for calibration intuition, not exact ICM replication
    """)