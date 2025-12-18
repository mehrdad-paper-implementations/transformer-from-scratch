"""
Transformer Encoder Layer
"""
from torch import nn
from multi_head_attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_v=None, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.d_k = d_model // n_heads
        self.d_v = d_v if d_v is not None else self.d_k

        self.multi_head_attention = MultiHeadAttention(self.n_heads, self.d_model, d_v)

        self.intermediate_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward_module = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True))

    def forward(self, x, padding_mask=None):
        """
        x_shape: [B, T, D]
        padding_mask: [B, T]
        """

        # Apply multi-head-attention
        output = self.multi_head_attention(x, x, x, padding_mask)  # [B, T, D]

        # Add and norm
        output = self.dropout(output)
        normed_output = self.intermediate_layer_norm(x + output)

        # Feed-Forward
        output = self.feed_forward_module(normed_output)

        # Add and norm
        normed_output = self.final_layer_norm(output + normed_output)

        return normed_output
