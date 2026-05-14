#!/usr/bin/env python3
"""
data_download.py: Download USGS ComCat Indonesia earthquake catalog (2010-2026) as CSV.
"""
import logging
from pathlib import Path
import requests
import pandas as pd
from io import StringIO

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1"

# Query parameters
START_YEAR = 2010
END_YEAR = 2026
MIN_MAG = 4.5
MIN_LATITUDE = -11.0
MAX_LATITUDE = 6.0
MIN_LONGITUDE = 95.0
MAX_LONGITUDE = 141.0

def fetch_yearly_csv(year: int) -> pd.DataFrame:
    """
    Fetch earthquake data for a given year via USGS FDSN API.
    """
    starttime = f"{year}-01-01"
    endtime = f"{year}-12-31"
    params = {
        "format": "csv",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": MIN_MAG,
        "minlatitude": MIN_LATITUDE,
        "maxlatitude": MAX_LATITUDE,
        "minlongitude": MIN_LONGITUDE,
        "maxlongitude": MAX_LONGITUDE,
    }
    logging.info(f"Fetching {year} data from USGS API...")
    url = f"{USGS_API_URL}/query"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    all_dfs = []
    # Loop by year to avoid query limits
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            df_year = fetch_yearly_csv(year)
        except Exception as e:
            logging.error(f"Failed to fetch data for {year}: {e}")
            continue
        count = len(df_year)
        logging.info(f"Year {year}: retrieved {count} events")
        all_dfs.append(df_year)
    if not all_dfs:
        logging.error("No data fetched. Exiting.")
        return
    # Concatenate all dataframes
    data_all = pd.concat(all_dfs, ignore_index=True)
    # Drop duplicates if any
    data_all = data_all.drop_duplicates(subset=["id"])
    # Save raw CSV
    output_file = RAW_DIR / "indonesia_raw.csv"
    data_all.to_csv(output_file, index=False)
    logging.info(f"Saved raw data to {output_file} ({len(data_all)} events total)")

if __name__ == "__main__":
    main()
