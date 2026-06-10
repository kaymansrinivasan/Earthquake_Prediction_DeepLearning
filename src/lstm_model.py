import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_zones=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc_mag = nn.Linear(hidden_dim, 1)
        self.fc_zone = nn.Linear(hidden_dim, num_zones)
        self.fc_large = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        out, (hidden, cell) = self.lstm(x)

        h = self.dropout(hidden[-1])

        mag = self.fc_mag(h)
        zone = self.fc_zone(h)
        large = self.fc_large(h)

        return mag, zone, large