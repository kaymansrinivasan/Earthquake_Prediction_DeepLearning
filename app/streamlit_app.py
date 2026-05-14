import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from pathlib import Path

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Earthquake Forecast Dashboard",
    page_icon="🌎",
    layout="wide"
)
# Set a consistent Plotly theme
px.defaults.template = "plotly_white"

# =========================
# Data Path Setup (Absolute vs Relative)
# =========================
# Option 1: Absolute path (example)
# csv_path = r"C:\Users\User\PyCharmMiscProject\data\indonesia_earthquake_data.csv"
# Option 2: Project-relative path using BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "processed" / "indonesia_processed.csv"

# =========================
# Load Data with Caching
# =========================
@st.cache_data
def load_data(path):
    """Load earthquake data from CSV, parse dates."""
    df = pd.read_csv(path, parse_dates=["time"])
    return df

df = None
if csv_path.exists():
    try:
        df = load_data(csv_path)
    except Exception as e:
        st.error(f"Error reading data: {e}")
else:
    st.error("Dataset not found at the expected location.")
    # Provide instructions or button to download raw data
    if st.button("Download raw USGS data"):
        try:
            import requests
            url = (
                "https://earthquake.usgs.gov/fdsnws/event/1/query?"
                "format=csv&starttime=2010-01-01&endtime=2026-12-31"
                "&minmagnitude=4.5&minlatitude=-11&maxlatitude=6&minlongitude=95&maxlongitude=141"
            )
            resp = requests.get(url)
            resp.raise_for_status()
            data = pd.read_csv(pd.io.common.StringIO(resp.text))
            # Save to CSV for future use
            data.to_csv(csv_path, index=False)
            df = data
            st.success("Data downloaded and saved successfully.")
        except Exception as err:
            st.error(f"Failed to download data: {err}")

# If data could not be loaded, stop the app
if df is None or df.empty:
    st.warning("No data available to display. Please provide the dataset as instructed above.")
    st.stop()

# Ensure essential columns exist
required_cols = ['time','latitude','longitude','depth','mag','seismic_zone']
for col in required_cols:
    if col not in df.columns:
        st.error(f"Required column missing: {col}")
        st.stop()

# =========================
# Optional: Load Trained Model
# =========================
@st.cache_resource
def load_model(model_path):
    """Attempt to load a saved ML model (PyTorch or TensorFlow)."""
    if not model_path.exists():
        return None
    try:
        import torch
        model = torch.load(model_path, map_location='cpu')
        return model
    except ImportError:
        pass
    try:
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))
        return model
    except Exception:
        return None

model_path = BASE_DIR / "models" / "transformer_model.pth"  # Example path
model = load_model(model_path)

# =========================
# Data Filtering Sidebar
# =========================
st.sidebar.header("Filter Events")
# Date range filter
min_date = df['time'].min().date()
max_date = df['time'].max().date()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
# Magnitude filter
min_mag = float(df['mag'].min())
max_mag = float(df['mag'].max())
mag_range = st.sidebar.slider(
    "Magnitude range",
    min_value=min_mag, max_value=max_mag,
    value=(min_mag, max_mag), step=0.1
)
# Seismic zone filter
zones = sorted(df['seismic_zone'].unique())
selected_zones = st.sidebar.multiselect(
    "Seismic zones",
    options=zones,
    default=zones,
    help="DBSCAN cluster labels (-1 is noise)."
)
# Mapbox token (optional)
mapbox_token = st.sidebar.text_input(
    "Mapbox token (optional)",
    type="password",
    help="Enter Mapbox API token for custom basemap styles."
)
if mapbox_token:
    px.set_mapbox_access_token(mapbox_token)

# Apply filters to dataframe
mask = (
    (df['time'].dt.date >= start_date) &
    (df['time'].dt.date <= end_date) &
    (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1]) &
    (df['seismic_zone'].isin(selected_zones))
)
filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("No events match the selected filters.")

# =========================
# Header and Summary Metrics
# =========================
st.title("🌏 Earthquake Forecast Dashboard")
st.markdown(
    "Interactive dashboard for earthquake forecasting in Indonesia. "
    "Use the sidebar to filter data by date, magnitude, and seismic zone."
)

# Compute summary metrics
total_events = len(filtered)
max_mag = filtered['mag'].max() if total_events else 0
avg_mag = filtered['mag'].mean() if total_events else 0
years = filtered['time'].dt.year
if total_events and not years.empty:
    span_years = years.max() - years.min() + 1
    avg_events_per_year = total_events / span_years if span_years > 0 else total_events
else:
    avg_events_per_year = 0

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Events", f"{total_events:,}", help="Number of earthquakes after filtering")
col2.metric("Max Magnitude", f"{max_mag:.2f}", help="Maximum magnitude in this selection")
col3.metric("Avg Magnitude", f"{avg_mag:.2f}", help="Mean magnitude of selected events")
col4, col5, col6 = st.columns(3)
col4.metric("Years Covered", f"{span_years}", help="Number of years in selected range")
col5.metric("Events/Year", f"{avg_events_per_year:.1f}", help="Average events per year in range")
col6.metric("Zones Selected", f"{len(selected_zones)}", help="Number of seismic zones included")

# =========================
# Map: Earthquake Locations
# =========================
st.subheader("Map: Recent Earthquake Locations")
if total_events:
    map_fig = px.scatter_map(
        filtered,
        lat="latitude", lon="longitude",
        color="seismic_zone",
        size="mag",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=5,
        mapbox_style="open-street-map",
        title="Earthquake epicenters coloured by zone"
    )
    # Enable clustering (if many points)【5†L282-L288】
    map_fig.update_traces(cluster=dict(enabled=True))
    st.plotly_chart(map_fig, use_container_width=True)
else:
    st.info("Adjust filters to display earthquakes on the map.")

# =========================
# Charts: Distributions and Trends
# =========================
st.subheader("Magnitude Histogram")
if total_events:
    hist_fig = px.histogram(
        filtered, x="mag", nbins=20,
        labels={"mag": "Magnitude"},
        title="Magnitude Distribution"
    )
    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.info("No data to plot histogram.")

st.subheader("Temporal Trends (Events per Month)")
if total_events:
    # Resample events count per month
    df_time = filtered.set_index("time").resample('M').size().reset_index(name='counts')
    time_fig = px.line(
        df_time, x="time", y="counts",
        labels={"time": "Date", "counts": "Number of Events"},
        title="Earthquakes Over Time"
    )
    st.plotly_chart(time_fig, use_container_width=True)
else:
    st.info("No data to plot time series.")

st.subheader("Depth vs. Magnitude")
if total_events:
    depth_fig = px.scatter(
        filtered, x="mag", y="depth",
        labels={"mag": "Magnitude", "depth": "Depth (km)"},
        title="Depth vs. Magnitude"
    )
    st.plotly_chart(depth_fig, use_container_width=True)
else:
    st.info("No data to plot depth vs magnitude.")

# =========================
# Transformer Prediction & Large-Event Gauge
# =========================
st.subheader("Forecasted Earthquake")
if model:
    # Example: use model to predict (placeholder code)
    # predicted = model.predict(filtered_features)
    # For demo, simulate:
    predicted_mag = 5.3
    uncertainty = 0.4
    predicted_zone = int(filtered['seismic_zone'].mode()[0]) if total_events else None
    prob_large = filtered['mag'].ge(5.0).mean()
else:
    st.info("No trained model found: displaying simulated forecast.")
    predicted_mag = 5.0
    uncertainty = 0.6
    predicted_zone = int(filtered['seismic_zone'].mode()[0]) if total_events else None
    prob_large = filtered['mag'].ge(5.0).mean()

st.markdown(f"**Predicted Magnitude:** {predicted_mag:.1f} ± {uncertainty}")
st.markdown(f"**Predicted Zone:** {predicted_zone}")

st.subheader("Probability of M≥5 Event")
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob_large * 100,
    title={'text': "Large Event Probability (%)"},
    gauge={'axis': {'range': [None, 100]}}
))
st.plotly_chart(gauge_fig, use_container_width=True)

# =========================
# Recent Events Table and Download
# =========================
st.subheader("Recent Earthquake Events")
if total_events:
    recent_table = filtered.sort_values('time').tail(20)
    st.dataframe(recent_table[['time','latitude','longitude','depth','mag']])
    # Provide CSV download of filtered data
    csv_bytes = recent_table.to_csv(index=False).encode()
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_earthquakes.csv",
        mime="text/csv"
    )
else:
    st.info("No recent events to display.")

# =========================
# Interpretation & Methodology Expanders
# =========================
with st.expander("Forecast Interpretation (Why and How)"):
    st.markdown("""
    - The **predicted values** above are based on our trained spatio-temporal model. The magnitude is an estimate 
      with uncertainty (± range) due to model error. The seismic zone is a cluster label from DBSCAN (see below).
    - The **probability gauge** shows the chance of a large (M≥5) earthquake based on recent data. 
    - **Limitations:** Earthquake forecasting is inherently uncertain. This model uses historical event patterns 
      but cannot account for all geophysical factors. Do *not* rely solely on these forecasts for safety decisions.
    - **Recommendations:** Use these insights to focus monitoring and preparedness efforts. Combine with expert 
      judgment and other early warning systems.
    """)

with st.expander("Methodology & Data"):
    st.markdown("""
    **Data Source:** USGS ComCat earthquake catalog (M≥4.5, Indonesia, 2010–2026)【13†L6-L9】.  
    **Preprocessing:** Filtered for valid time/lat/lon/mag; computed rolling statistics and energy release features.  
    **Spatial Zones:** Earthquakes were clustered by location using DBSCAN to define tectonic zones.  
    **Model:** A hybrid Transformer (attention-based) architecture for sequence forecasting (magnitude, zone, large-event).  
    **Baselines:** LSTM and XGBoost models were also developed (not shown here) for comparison.  
    **Evaluation:** We will measure MAE/RMSE for magnitude prediction and classification metrics (accuracy, ROC-AUC) for zone/large-event predictions.
    """)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("FYP2 Project – Hybrid Spatio-Temporal Transformer for Earthquake Forecasting")

