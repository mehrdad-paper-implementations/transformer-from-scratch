"""
Transformer Decoder Block
"""
from torch import nn
from transformer_decoder_layer import TransformerDecoderLayer


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, d_v: int, dropout: float, n_layers: int):
        super().__init__()

        self.transformer_block = nn.ModuleList([
            TransformerDecoderLayer(n_heads, d_model, d_ff, d_v, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output, causal_mask, padding_mask=None):
        """
        x_shape: [B, T_dec, D]
        encoder_output_shape: [B, T_enc, D]
        causal_mask: [B, T_dec, T_dec]
        padding_mask: [B, T_dec]
        """
        output = x
        for layer in self.transformer_block:
            output = layer(output, encoder_output, padding_mask=padding_mask, causal_mask=causal_mask)
        return output
