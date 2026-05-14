# Earthquake Forecasting Project

This repository implements an end-to-end earthquake forecasting pipeline using a hybrid spatio-temporal Transformer model. The workflow includes:

- **Data Download**: Query USGS ComCat API for Indonesia earthquakes (2010–2026, M≥4.5) and save raw CSV data.
- **Preprocessing**: Clean and enrich data with temporal features, rolling statistics, energy, and large-event labels.
- **Spatial Clustering**: Apply DBSCAN to latitude/longitude to assign seismic zone IDs.
- **Sequence Generation**: Build fixed-length event sequences for model input (multi-task targets: magnitude, zone, large-event).
- **Model**: PyTorch Transformer with separate heads for magnitude (regression), zone (classification), and large-event (probability).
- **Training & Evaluation**: Train model, compute MAE/RMSE (magnitude), AUC (large-event), and accuracy (zone).
- **Inference & Dashboard**: Run model inference and visualize results in a Streamlit dashboard (KPIs, maps, charts).

## Requirements

See [requirements.txt](requirements.txt) for dependencies. Install via:

