"""
Transformer Decoder Layer
"""
from torch import nn
from multi_head_attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_v=None, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.d_k = d_model // n_heads
        self.d_v = d_v if d_v is not None else self.d_k

        self.multi_head_attention = MultiHeadAttention(self.n_heads, self.d_model, d_v)
        self.masked_multi_head_attention = MultiHeadAttention(self.n_heads, self.d_model, d_v)

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.enc_dec_attention_layer_norm = nn.LayerNorm(d_model)
        self.feedforward_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward_module = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True))

    def forward(self, x, encoder_output, padding_mask=None, causal_mask=None):
        """
        x_shape: [B, T, d_v]
        q_shape: [B, T, D]
        k_shape: [B, T, D]
        padding_mask: [B, T]
        causal_mask: [B, T, T]
        """

        mask = causal_mask
        if padding_mask is not None:
            mask = mask & padding_mask.unsqueeze(1)

        # Apply masked multi-head-attention
        output = self.masked_multi_head_attention(x, x, x, mask)  # [B, T, D]

        # Add and norm
        output = self.dropout(output)
        temp_output = self.self_attention_layer_norm(x + output)

        output = self.multi_head_attention(q=temp_output, k=encoder_output, v=encoder_output, mask=padding_mask)

        # Add and norm
        output = self.dropout(output)
        temp_output = self.enc_dec_attention_layer_norm(output + temp_output)

        # Feed-Forward
        output = self.feed_forward_module(temp_output)

        # Add and norm
        normed_output = self.feedforward_layer_norm(output + temp_output)

        return normed_output
