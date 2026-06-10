import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
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
# Days since previous earthquake
df['days_since_last_quake'] = (
    df['time']
    .diff()
    .dt.total_seconds()
    .div(86400)
    .fillna(0)
)

# Magnitude change from previous event
df['mag_change'] = df['mag'].diff().fillna(0)

# Depth change from previous event
df['depth_change'] = df['depth'].diff().fillna(0)

# -----------------------------
# Rolling Statistics
# -----------------------------
window = 30

df['rolling_mag_mean'] = df['mag'].rolling(window).mean()
df['rolling_mag_std'] = df['mag'].rolling(window).std()
df['rolling_mag_max'] = df['mag'].rolling(window).max()

df['rolling_depth_mean'] = df['depth'].rolling(window).mean()
df['rolling_depth_std'] = df['depth'].rolling(window).std()



# -----------------------------
# Energy Release Feature
# -----------------------------
df['energy_release'] = 10 ** (1.5 * df['mag'] + 4.8)

df['rolling_energy_mean'] = (
    df['energy_release']
    .rolling(window)
    .mean()
)

# -----------------------------
# Large Event Classification
# -----------------------------
df['large_event'] = (df['mag'] >= 5.0).astype(int)
print("\nLarge Event Distribution:")
print(df['large_event'].value_counts())
print(df['large_event'].value_counts(normalize=True))

# -----------------------------
# kmeans Seismic Zoning
# -----------------------------
coords = df[['latitude', 'longitude']].values

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['seismic_zone'] = kmeans.fit_predict(coords)

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
print(df.columns.tolist())