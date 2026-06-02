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
    duration_hr = st.slider("Storm duration (hr)", 0.25, 24.0, 1.00)
    peak_intensity_mmhr = st.slider("Peak intensity (mm/hr)", 1.0, 100.0, 20.0)
    start_hr = st.slider("Storm start time (hr)", 0.0, 24.0, 4.0)
    total_sim_hr = st.slider("Simulation length (hr)", 6.0, 168.0, 48.0)
    dt_min = st.slider("Time step (min)", 0.2, 5.0, 1.0)

    st.markdown("---")

    st.header("Surface routing (simplified)")

    runoff_fraction = st.slider("Runoff fraction (PR)", 0.0, 1.0, 0.0)
    contributing_area_pct = st.slider(
        "Proportion of area contributing to soil store (%)",
        0.0, 100.0, 50.0
    )
    depression_storage_mm = st.slider("Depression storage (mm)", 0.0, 20.0, 2.0)

    st.markdown("---")

    st.header("Soil store parameters")

    soil_depth_m = st.slider("Soil depth (m)", 0.1, 5.0, 1.0)
    soil_porosity_pct = st.slider("Soil porosity (%)", 1.0, 100.0, 50.0)
    percolation_threshold_pct = st.slider("Percolation threshold (%)", 0.0, 100.0, 10.5)
    percolation_coefficient = st.slider("Percolation coefficient (1/day)", 0.1, 10.0, 1.0)
    percolation_percentage = st.slider("Percolation % infiltrating", 0.0, 100.0, 50.0)
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

    dt_s = dt_min * 60.0
    dt_hr = dt_min / 60.0

    t = np.arange(0, total_sim_hr, dt_hr)
    rain_mmhr = make_storm(t, storm_type, start_hr, duration_hr, peak_intensity_mmhr)

    n = len(t)

    # ----------------------------
    # States
    # ----------------------------
    D = np.zeros(n)          # soil water depth [m]
    dep_store = 0.0          # depression storage currently filled [m]

    # ----------------------------
    # Flows
    # ----------------------------
    q_perc = np.zeros(n)     # percolation from soil store [m3/s]
    q_ri = np.zeros(n)       # rainfall-induced infiltration to sewer [m3/s]
    q_soil = np.zeros(n)     # inflow to soil store [m3/s]
    q_runoff = np.zeros(n)   # simplified runoff [m3/s]
    q_evap = np.zeros(n)     # evaporation from soil store [m3/s]

    # diagnostics
    x_excess = np.zeros(n)   # soil depth above threshold [m]

    # ----------------------------
    # Parameters
    # ----------------------------
    Psoil = soil_porosity_pct / 100.0
    alpha = percolation_percentage / 100.0
    area_factor = contributing_area_pct / 100.0

    Dmax = soil_depth_m
    Dt = (percolation_threshold_pct / 100.0) * Dmax

    # Percolation coefficient as timescale in days
    tau_s = percolation_coefficient * 86400.0
    k_eff = 1.0 / tau_s if tau_s > 0 else 0.0  # 1/s

    evap_full = potential_evap_mmday / 1000.0 / 86400.0  # m/s

    D[0] = np.clip((initial_saturation / 100.0) * Dmax, 0.0, Dmax)
    dep_store_max = depression_storage_mm / 1000.0

    for i in range(1, n):
        rain_mps = rain_mmhr[i] / 1000.0 / 3600.0
        rainfall_depth = rain_mps * dt_s  # [m] over timestep

        # ----------------------------------------
        # 1) Depression storage
        # ----------------------------------------
        available = max(dep_store_max - dep_store, 0.0)
        fill = min(rainfall_depth, available)
        dep_store += fill
        effective = rainfall_depth - fill

        # ----------------------------------------
        # 2) Simplified partitioning
        # ----------------------------------------
        soil_depth_input = (1.0 - runoff_fraction) * area_factor * effective
        runoff_depth = (
            runoff_fraction * effective +
            (1.0 - area_factor) * effective
        )

        q_soil[i] = soil_depth_input * A / dt_s
        q_runoff[i] = runoff_depth * A / dt_s

        # ----------------------------------------
        # 3) Soil evaporation
        # ----------------------------------------
        sat = D[i - 1] / Dmax if Dmax > 0 else 0.0
        q_evap[i] = evap_full * sat * A

        # If no soil porosity, just zero everything safely
        if Psoil <= 0:
            D[i] = 0.0
            q_perc[i] = 0.0
            q_ri[i] = 0.0
            x_excess[i] = 0.0
            continue


        # ----------------------------------------
        # 4) Work in storage-above-threshold space
        # ----------------------------------------

        # current excess above threshold
        x_old = max(D[i - 1] - Dt, 0.0)

        # net forcing into soil store as depth rate [m/s]
        # (this already includes evap correctly)
        u = (q_soil[i] - q_evap[i]) / (Psoil * A) if Psoil > 0 else 0.0

        # --- Predictor ONLY for detecting threshold crossing ---
        D_pred = D[i - 1] + u * dt_s
        D_pred = np.clip(D_pred, 0.0, Dmax)

        # ----------------------------------------
        # PERMEATION / PERCOLATION CALCULATION
        # ----------------------------------------
        if D_pred <= Dt:
            # below threshold → no percolation
            q_perc[i] = 0.0
            x_excess[i] = 0.0

        else:
            # --- Now integrate ABOVE-THRESHOLD reservoir ONLY ---
            # IMPORTANT: use x_old (actual state), not D_pred (fixes peak behaviour)

            if k_eff > 0.0:
                expfac = np.exp(-k_eff * dt_s)

                # driven linear reservoir update (smooths peak properly)
                x_new = x_old * expfac + (u / k_eff) * (1.0 - expfac)
            else:
                x_new = x_old + u * dt_s

            # prevent negative excess
            x_new = max(x_new, 0.0)

            # compute percolation from UPDATED excess
            q_perc[i] = k_eff * A * x_new
            x_excess[i] = x_new

        # rainfall-induced infiltration (clean version)
        q_ri[i] = alpha * q_perc[i]

        # ----------------------------------------
        # FULL SOIL STORE MASS BALANCE (ALWAYS APPLY)
        # ----------------------------------------
        if Psoil > 0:
            dDdt = (q_soil[i] - q_evap[i] - q_perc[i]) / (Psoil * A)
            D[i] = np.clip(D[i - 1] + dDdt * dt_s, 0.0, Dmax)
        else:
            D[i] = 0.0


        # ----------------------------------------
        # 5) Cap at Dmax if needed
        # ----------------------------------------
        if D[i] > Dmax:
            D[i] = Dmax
            x_excess[i] = max(Dmax - Dt, 0.0)
            q_perc[i] = k_eff * A * x_excess[i]
            q_ri[i] = alpha * q_perc[i]

    df = pd.DataFrame({
        "time_hr": t,
        "rain": rain_mmhr,
        "soil_depth": D,
        "saturation_pct": 100.0 * D / Dmax if Dmax > 0 else np.zeros_like(D),
        "soil_excess_above_threshold_m": x_excess,
        "soil_inflow_lps": q_soil * 1000.0,
        "percolation_lps": q_perc * 1000.0,
        "ri_infiltration_lps": q_ri * 1000.0,
        "runoff_lps": q_runoff * 1000.0,
        "evap_lps": q_evap * 1000.0,
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
fig.add_scatter(x=df.time_hr, y=df.soil_inflow_lps, name="Soil inflow", visible='legendonly')
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