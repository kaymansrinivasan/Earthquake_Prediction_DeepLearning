#!/usr/bin/env python3
"""
spatial.py: Perform DBSCAN clustering on earthquake locations to assign seismic zones.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Configuration
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "indonesia_processed.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZONE_CSV = OUTPUT_DIR / "indonesia_zones.csv"

# DBSCAN parameters
EPS_KM = 50.0  # cluster radius in kilometers
MIN_SAMPLES = 20

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        df = pd.read_csv(PROCESSED_CSV)
    except FileNotFoundError as e:
        logging.error(f"Processed data file not found: {e}")
        return

    coords = df[['latitude', 'longitude']].to_numpy()
    # Convert lat/lon to radians for haversine metric
    coords_rad = np.radians(coords)
    # Compute DBSCAN eps in radians
    earth_radius_km = 6371.0
    eps_rad = EPS_KM / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine')
    labels = db.fit_predict(coords_rad)
    df['zone'] = labels

    n_zones = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f"Identified {n_zones} seismic zone(s) (DBSCAN clusters)")

    # Save mapping of event id to zone label
    if 'id' in df.columns:
        mapping_df = df[['id', 'zone']]
        mapping_df.to_csv(ZONE_CSV, index=False)
        logging.info(f"Saved zone mapping to {ZONE_CSV}")
    else:
        logging.warning("'id' column not found; skipping zone mapping output")

if __name__ == "__main__":
    main()
