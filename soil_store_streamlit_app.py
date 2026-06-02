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
    A = 10000.0  # 1 hectare total area

    dt_s = dt_min * 60
    dt_hr = dt_min / 60

    t = np.arange(0, total_sim_hr, dt_hr)
    rain_mmhr = make_storm(t, storm_type, start_hr, duration_hr, peak_intensity_mmhr)

    n = len(t)

    # states
    D = np.zeros(n)                # soil water depth [m]
    dep_store = np.zeros(n)        # depression storage depth [m]

    # flows
    q_perc = np.zeros(n)           # percolation from soil store [m3/s]
    q_ri = np.zeros(n)             # rainfall-induced infiltration to network [m3/s]
    q_soil = np.zeros(n)           # inflow to soil store [m3/s]
    q_runoff = np.zeros(n)         # surface runoff to network / bypass [m3/s]
    q_dep_evap = np.zeros(n)       # evaporation from depression storage [m3/s]
    q_soil_evap = np.zeros(n)      # evaporation from soil store [m3/s]
    q_overflow = np.zeros(n)       # overflow when soil store full [m3/s]

    # parameters
    Psoil = soil_porosity_pct / 100.0
    alpha = percolation_percentage / 100.0
    area_factor = contributing_area_pct / 100.0

    Dmax = soil_depth_m
    Dt = (percolation_threshold_pct / 100.0) * Dmax

    # Percolation coefficient is treated as a time coefficient in days:
    # larger value => slower / longer-duration response
    tau_s = percolation_coefficient * 86400.0
    k_eff = 1.0 / tau_s

    evap_mps = potential_evap_mmday / 1000.0 / 86400.0

    # Initial condition
    D[0] = min((initial_saturation / 100.0) * Dmax, Dmax)

    dep_store_max = depression_storage_mm / 1000.0

    for i in range(1, n):
        rain_mps = rain_mmhr[i] / 1000.0 / 3600.0
        rain_depth = rain_mps * dt_s  # rainfall depth this step [m]

        # ------------------------------------------------------------
        # 1) Depression storage first (ICM-like sequencing)
        # ------------------------------------------------------------

        # evaporation loss from depression storage
        dep_evap_depth = min(dep_store[i - 1], evap_mps * dt_s)
        dep_store_tmp = dep_store[i - 1] - dep_evap_depth
        q_dep_evap[i] = dep_evap_depth * A / dt_s

        # fill depression storage
        dep_available = max(dep_store_max - dep_store_tmp, 0.0)
        dep_fill = min(rain_depth, dep_available)
        dep_store_after_fill = dep_store_tmp + dep_fill

        # effective rainfall after depression storage
        effective_depth = rain_depth - dep_fill

        # ------------------------------------------------------------
        # 2) Partition effective rainfall:
        #    - non-contributing area => runoff
        #    - contributing area split by PR into runoff vs soil-store inflow
        # ------------------------------------------------------------
        soil_candidate_depth = effective_depth * area_factor * (1.0 - runoff_fraction)
        runoff_depth = (
            effective_depth * (1.0 - area_factor) +
            effective_depth * area_factor * runoff_fraction
        )

        # ------------------------------------------------------------
        # 3) Soil store evaporation and percolation
        # ------------------------------------------------------------
        sat = D[i - 1] / Dmax if Dmax > 0 else 0.0

        soil_evap_depth = min(D[i - 1], evap_mps * sat * dt_s)
        q_soil_evap[i] = soil_evap_depth * Psoil * A / dt_s

        if D[i - 1] < Dt:
            q_perc[i] = 0.0
        else:
            q_perc[i] = k_eff * A * max(D[i - 1] - Dt, 0.0)

        q_ri[i] = alpha * q_perc[i] * Psoil

        # ------------------------------------------------------------
        # 4) Update soil store with overflow routed to runoff
        #    (instead of clipping away mass)
        # ------------------------------------------------------------
        net_soil_depth_change = (
            soil_candidate_depth
            - soil_evap_depth
            - (q_perc[i] * dt_s) / (Psoil * A) if Psoil > 0 else 0.0
        )

        D_trial = D[i - 1] + net_soil_depth_change

        if D_trial > Dmax:
            overflow_depth = D_trial - Dmax
            D[i] = Dmax
        elif D_trial < 0:
            overflow_depth = 0.0
            D[i] = 0.0
        else:
            overflow_depth = 0.0
            D[i] = D_trial

        # Soil overflow goes to runoff (closer to ICM behaviour)
        runoff_depth += overflow_depth
        q_overflow[i] = overflow_depth * A / dt_s

        q_soil[i] = soil_candidate_depth * A / dt_s
        q_runoff[i] = runoff_depth * A / dt_s
        dep_store[i] = dep_store_after_fill

    df = pd.DataFrame({
        "time_hr": t,
        "rain": rain_mmhr,
        "soil_depth": D,
        "saturation_pct": 100.0 * D / Dmax if Dmax > 0 else 0.0,
        "depression_storage_mm": dep_store * 1000.0,
        "soil_inflow_lps": q_soil * 1000.0,
        "percolation_lps": q_perc * 1000.0,
        "ri_infiltration_lps": q_ri * 1000.0,
        "surface_runoff_lps": q_runoff * 1000.0,
        "soil_overflow_to_runoff_lps": q_overflow * 1000.0,
        "dep_storage_evap_lps": q_dep_evap * 1000.0,
        "soil_evap_lps": q_soil_evap * 1000.0,
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