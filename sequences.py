#!/usr/bin/env python3
"""
sequences.py: Create fixed-length sequences for earthquake forecasting and wrap in a PyTorch Dataset.
"""
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EarthquakeDataset(Dataset):
    """
    PyTorch Dataset for earthquake sequences. Each sample contains a sequence of past events and multi-task targets.
    """
    def __init__(self, csv_file: Path, window: int, stride: int = 1, feature_cols: List[str] = None,
                 zone_col: str = 'zone', mag_col: str = 'mag', large_col: str = 'large_event'):
        """
        Args:
            csv_file: Path to processed CSV with features and labels.
            window: number of past events in each input sequence.
            stride: step size between sequence windows.
            feature_cols: list of column names to use as features.
            zone_col: column name for seismic zone label.
            mag_col: column name for magnitude target.
            large_col: column name for large-event binary target.
        """
        df = pd.read_csv(csv_file)
        if feature_cols is None:
            # Default features: latitude, longitude, depth, magnitude, time difference
            feature_cols = ['latitude', 'longitude', 'depth', 'mag', 'time_diff_hours']
        self.feature_cols = feature_cols

        self.X = []
        self.y_mag = []
        self.y_zone = []
        self.y_large = []
        data = df.reset_index(drop=True)
        total_len = len(data)
        for start in range(0, total_len - window, stride):
            end = start + window
            seq = data.loc[start:end-1, feature_cols].values.astype(np.float32)
            if np.isnan(seq).any():
                continue
            target = data.loc[end]
            # Skip if any targets are NaN
            if pd.isna(target[mag_col]) or pd.isna(target[zone_col]) or pd.isna(target[large_col]):
                continue
            self.X.append(seq)
            self.y_mag.append(target[mag_col])
            self.y_zone.append(int(target[zone_col]) if zone_col in data.columns else 0)
            self.y_large.append(float(target[large_col]))

        self.X = torch.tensor(self.X)  # shape: (N, window, features)
        self.y_mag = torch.tensor(self.y_mag).unsqueeze(1)  # shape: (N, 1)
        self.y_zone = torch.tensor(self.y_zone, dtype=torch.long)  # shape: (N,)
        self.y_large = torch.tensor(self.y_large).unsqueeze(1)  # shape: (N, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_mag[idx], self.y_zone[idx], self.y_large[idx]
