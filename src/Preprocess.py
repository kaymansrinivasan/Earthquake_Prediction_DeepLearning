import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "indonesia_raw.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "indonesia_processed.csv"

# Load raw dataset
df = pd.read_csv(RAW_DATA_PATH)

# -----------------------------
# Basic Cleaning
# -----------------------------
required_cols = ['time', 'latitude', 'longitude', 'depth', 'mag']

df = df.dropna(subset=required_cols)

df['time'] = pd.to_datetime(df['time'])

df = df.sort_values('time').reset_index(drop=True)

# -----------------------------
# Temporal Features
# -----------------------------
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

df['time_diff_hours'] = (
    df['time']
    .diff()
    .dt.total_seconds()
    .div(3600)
    .fillna(0)
)

# -----------------------------
# Rolling Statistics
# -----------------------------
window = 30

df['rolling_mag_mean'] = df['mag'].rolling(window).mean()
df['rolling_mag_std'] = df['mag'].rolling(window).std()
df['rolling_mag_max'] = df['mag'].rolling(window).max()

# -----------------------------
# Energy Release Feature
# -----------------------------
df['energy_release'] = 10 ** (1.5 * df['mag'])

# -----------------------------
# Large Event Classification
# -----------------------------
df['large_event'] = (df['mag'] >= 5.0).astype(int)

# -----------------------------
# DBSCAN Seismic Zoning
# -----------------------------
coords = df[['latitude', 'longitude']].values

coords_rad = np.radians(coords)

earth_radius_km = 6371.0
eps_km = 150

eps_rad = eps_km / earth_radius_km

dbscan = DBSCAN(
    eps=eps_rad,
    min_samples=20,
    metric='haversine'
)

df['seismic_zone'] = dbscan.fit_predict(coords_rad)

# -----------------------------
# Remove NaNs after rolling ops
# -----------------------------
df = df.dropna().reset_index(drop=True)

# -----------------------------
# Save Processed Dataset
# -----------------------------
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED_DATA_PATH, index=False)

print("Processed dataset saved:")
print(PROCESSED_DATA_PATH)

print(df.head())

print(df['seismic_zone'].value_counts())