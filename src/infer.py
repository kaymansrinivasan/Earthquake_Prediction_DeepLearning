#!/usr/bin/env python3
"""
infer.py: Load the trained model checkpoint and run inference with optional uncertainty estimation.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sequences import EarthquakeDataset
from model import TransformerModel

# Configuration
SEED = 42
BATCH_SIZE = 64

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_CSV = BASE_DIR / "data" / "processed" / "indonesia_processed.csv"
    ZONE_CSV = BASE_DIR / "data" / "processed" / "indonesia_zones.csv"
    MODEL_PATH = BASE_DIR / "models" / "transformer_model.pth"

    # Load and prepare dataset (merge zones as in training)
    df = pd.read_csv(PROCESSED_CSV)
    zones = pd.read_csv(ZONE_CSV) if ZONE_CSV.exists() else None
    if zones is not None and 'id' in zones.columns:
        df = df.merge(zones, on='id', how='left')
        df['zone'].fillna(-1, inplace=True)
    else:
        df['zone'] = 0
    merged_csv = BASE_DIR / "data" / "processed" / "indonesia_merged.csv"
    df.to_csv(merged_csv, index=False)

    dataset = EarthquakeDataset(merged_csv, window=30, stride=1)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    n_zones = int(df['zone'].max() + 1)
    model = TransformerModel(input_dim=len(dataset.feature_cols),
                             embed_dim=64, num_heads=4, num_layers=2,
                             dropout=0.1, num_zones=n_zones)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    # Inference
    with torch.no_grad():
        for X, _, _, _ in loader:
            X = X.to(device)
            mag_pred, zone_logits, large_logit = model(X)
            mag_pred = mag_pred.cpu().numpy().flatten()
            zone_pred = torch.argmax(zone_logits, dim=1).cpu().numpy()
            large_prob = torch.sigmoid(large_logit).cpu().numpy().flatten()
            for m, z, lp in zip(mag_pred, zone_pred, large_prob):
                predictions.append((m, z, lp))

    # Save predictions to CSV
    pred_df = pd.DataFrame(predictions, columns=['pred_mag', 'pred_zone', 'pred_large_prob'])
    output_csv = BASE_DIR / "outputs" / "predictions.csv"
    output_csv.parent.mkdir(exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    logging.info(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
