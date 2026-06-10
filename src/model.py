#!/usr/bin/env python3
"""
model.py: Define the hybrid spatio-temporal Transformer model for earthquake forecasting.
"""
from typing import Tuple
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor of same shape with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TransformerModel(nn.Module):
    """
    Hybrid spatio-temporal Transformer for earthquake forecasting.
    Multi-task outputs: magnitude (regression), zone (classification), large-event (probability).
    """
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, num_layers: int,
                 dropout: float, num_zones: int):
        super().__init__()
        # Input embedding
        self.input_linear = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Output heads
        self.fc_mag = nn.Linear(embed_dim, 1)
        self.fc_zone = nn.Linear(embed_dim, num_zones)
        self.fc_large = nn.Linear(embed_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tuple of:
              - magnitude prediction (batch_size, 1)
              - zone logits (batch_size, num_zones)
              - large-event logit (batch_size, 1)
        """
        # Embedding and positional encoding
        x = self.input_linear(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        # Transformer expects input shape (seq_len, batch, embed_dim)

        x = self.transformer_encoder(x)  # (seq_len, batch, embed_dim)

        # Use last time step's output as summary
        x_last = torch.mean(x, dim=1)  # (batch, embed_dim)
        x_last = self.norm(x_last)
        x_last = self.dropout(x_last)
        mag_pred = self.fc_mag(x_last)      # (batch, 1)
        zone_logits = self.fc_zone(x_last)  # (batch, num_zones)
        large_logit = self.fc_large(x_last) # (batch, 1)
        return mag_pred, zone_logits, large_logit
