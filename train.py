#!/usr/bin/env python3
"""
train.py: Train the Transformer model on earthquake sequences and evaluate performance.
"""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score

from sequences import EarthquakeDataset
from model import TransformerModel

# Example configuration
config = {
    'seed': 42,
    'batch_size': 32,
    'epochs': 10,
    'lr': 1e-4,
    'window': 30,
    'stride': 1,
    'dropout': 0.1,
    'embed_dim': 64,
    'n_heads': 4,
    'n_layers': 2,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROCESSED_CSV = BASE_DIR / "data" / "processed" / "indonesia_processed.csv"
    ZONE_CSV = BASE_DIR / "data" / "processed" / "indonesia_zones.csv"

    # Load processed data and zones, merge
    df = pd.read_csv(PROCESSED_CSV)
    zones = pd.read_csv(ZONE_CSV) if ZONE_CSV.exists() else None
    if zones is not None and 'id' in zones.columns:
        df = df.merge(zones, on='id', how='left')
        df['zone'].fillna(-1, inplace=True)
    else:
        df['zone'] = 0
    df = df.sort_values('time').reset_index(drop=True)
    # Save merged data temporarily
    merged_csv = BASE_DIR / "data" / "processed" / "indonesia_merged.csv"
    df.to_csv(merged_csv, index=False)

    # Create dataset and split into train/test
    dataset = EarthquakeDataset(merged_csv, window=config['window'], stride=config['stride'])
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # Model instantiation
    n_zones = int(df['zone'].max() + 1)
    model = TransformerModel(input_dim=len(dataset.feature_cols),
                             embed_dim=config['embed_dim'],
                             num_heads=config['n_heads'],
                             num_layers=config['n_layers'],
                             dropout=config['dropout'],
                             num_zones=n_zones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # Loss functions
    criterion_mag = nn.MSELoss()
    criterion_zone = nn.CrossEntropyLoss()
    criterion_large = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.0
        for X, y_mag, y_zone, y_large in train_loader:
            X, y_mag, y_zone, y_large = X.to(device), y_mag.to(device), y_zone.to(device), y_large.to(device)
            optimizer.zero_grad()
            mag_pred, zone_logits, large_logit = model(X)
            loss_mag = criterion_mag(mag_pred, y_mag)
            loss_zone = criterion_zone(zone_logits, y_zone)
            loss_large = criterion_large(large_logit, y_large)
            loss = loss_mag + loss_zone + loss_large
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / train_size
        logging.info(f"Epoch {epoch}/{config['epochs']} - Training loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    true_mag, pred_mag = [], []
    true_zone, pred_zone = [], []
    true_large, pred_large = [], []
    with torch.no_grad():
        for X, y_mag, y_zone, y_large in test_loader:
            X = X.to(device)
            mag_pred, zone_logits, large_logit = model(X)
            pred_mag.extend(mag_pred.cpu().numpy().flatten().tolist())
            true_mag.extend(y_mag.numpy().flatten().tolist())
            pred_zone.extend(torch.argmax(zone_logits, dim=1).cpu().numpy().tolist())
            true_zone.extend(y_zone.numpy().tolist())
            pred_large.extend(torch.sigmoid(large_logit).cpu().numpy().flatten().tolist())
            true_large.extend(y_large.numpy().flatten().tolist())

    # Compute metrics
    mae = mean_absolute_error(true_mag, pred_mag)
    rmse = mean_squared_error(true_mag, pred_mag, squared=False)
    try:
        auc = roc_auc_score(true_large, pred_large)
    except ValueError:
        auc = float('nan')
    zone_acc = accuracy_score(true_zone, pred_zone)

    logging.info(f"Test MAE (magnitude): {mae:.4f}")
    logging.info(f"Test RMSE (magnitude): {rmse:.4f}")
    logging.info(f"Test ROC AUC (large event): {auc:.4f}")
    logging.info(f"Test Zone accuracy: {zone_acc:.4f}")

    # Save model checkpoint
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "transformer_model.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model checkpoint to {model_path}")

if __name__ == "__main__":
    main()
