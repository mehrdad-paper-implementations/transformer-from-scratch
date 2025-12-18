"""
Transformer Encoder Block
"""
from torch import nn
from .transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, d_v: int, dropout: float, n_layers: int):
        super().__init__()

        self.transformer_block = nn.ModuleList([
            TransformerEncoderLayer(n_heads, d_model, d_ff, d_v, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, padding_mask=None):
        """
        x_shape: [B, T, D]
        padding_mask: [B, T]
        """
        output = x
        for layer in self.transformer_block:
            output = layer(output, padding_mask=padding_mask)
        return output
