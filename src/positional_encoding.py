import math
import torch
from torch import nn


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # PE Matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute sin and cos for every dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to [1, T, D] to make it usable for batch
        pe = pe.unsqueeze(0)  # [1, T, D]

        # Register as buffer (This is not parameter and just loaded/saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
