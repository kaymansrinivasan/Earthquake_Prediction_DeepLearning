#!/usr/bin/env python3
"""
preprocess.py: Load raw CSV, clean, create features (time, rolling stats, energy, etc.), and save processed CSV.
"""
import logging
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "indonesia_raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_CSV = PROCESSED_DIR / "indonesia_processed.csv"

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        df = pd.read_csv(RAW_CSV)
    except FileNotFoundError as e:
        logging.error(f"Raw data file not found: {e}")
        return
    logging.info(f"Loaded raw data ({len(df)} records)")

    # Drop rows missing critical fields
    required_cols = ['time', 'latitude', 'longitude', 'depth', 'mag']
    df = df.dropna(subset=required_cols)
    logging.info(f"After dropping NaN in {required_cols}: {len(df)} records")

    # Parse time and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Temporal features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['time_diff_hours'] = df['time'].diff().dt.total_seconds() / 3600.0
    df['time_diff_hours'].fillna(0, inplace=True)

    # Rolling statistics on magnitude
    for window in [10, 30, 50]:
        df[f'mag_roll_mean_{window}'] = df['mag'].rolling(window, min_periods=1).mean()
        df[f'mag_roll_std_{window}'] = df['mag'].rolling(window, min_periods=1).std()

    # Compute energy release (log10 Joules) from magnitude
    df['energy_log10J'] = 1.5 * df['mag'] + 4.8

    # Label large events (M>=5.0)
    df['large_event'] = (df['mag'] >= 5.0).astype(int)

    # Save processed CSV
    df.to_csv(PROCESSED_CSV, index=False)
    logging.info(f"Saved processed data to {PROCESSED_CSV} ({len(df)} records)")

if __name__ == "__main__":
    main()
