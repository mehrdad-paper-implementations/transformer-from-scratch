import math
from torch import nn
from .transformer_encoder_block import TransformerEncoderBlock
from .transformer_decoder_block import TransformerDecoderBlock
from .positional_encoding import AbsolutePositionalEncoding


class Transformer(nn.Module):
    def __init__(self, n_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_model,
                 d_ff,
                 input_vocab_size,
                 output_vocab_size,
                 dropout=0.1,
                 d_emb=None,
                 n_dec_heads=None,
                 d_v=None):
        super().__init__()

        n_dec_heads = n_dec_heads if n_dec_heads is not None else n_heads
        self.d_model = d_model

        self.d_emb = d_emb if d_emb is not None else d_model

        if self.d_model != self.d_emb:
            self.decoder_project_emb_to_d_model = nn.Linear(d_emb, d_model)
            self.encoder_project_emb_to_d_model = nn.Linear(d_emb, d_model)
            self.project_d_model_to_emb = nn.Linear(d_model, d_emb)

        self.encoder_embedder = nn.Embedding(input_vocab_size, d_emb)
        self.decoder_embedder = nn.Embedding(output_vocab_size, d_emb)

        self.encoder_positional_encoding = AbsolutePositionalEncoding(d_model, dropout)
        self.decoder_positional_encoding = AbsolutePositionalEncoding(d_model, dropout)

        self.encoder = TransformerEncoderBlock(n_heads, d_model, d_ff, d_v, dropout, num_encoder_layers)
        self.decoder = TransformerDecoderBlock(n_dec_heads, d_model, d_ff, d_v, dropout, num_decoder_layers)

        self.project_to_vocab_size = nn.Linear(d_model, output_vocab_size, bias=False)
        # Share the decoder embedder weights with projection layer
        self.project_to_vocab_size.weight = self.decoder_embedder.weight

    def forward(self, input_ids, output_ids, causal_mask, input_mask=None, output_mask=None):
        x_emb = self.encoder_embedder(input_ids) / math.sqrt(self.d_emb)

        if self.d_model != self.d_emb:
            x_emb = self.encoder_project_emb_to_d_model(x_emb)

        x_emb = self.encoder_positional_encoding(x_emb)

        memory = self.encoder(x_emb, input_mask)

        y_emb = self.decoder_embedder(output_ids) / math.sqrt(self.d_emb)

        if self.d_model != self.d_emb:
            y_emb = self.decoder_project_emb_to_d_model(y_emb)

        y_emb = self.decoder_positional_encoding(y_emb)

        output = self.decoder(y_emb, memory, causal_mask, output_mask)

        if self.d_model != self.d_emb:
            output = self.project_d_model_to_emb(output)

        logits = self.project_to_vocab_size(output)  # [B, T_dec, D]

        return logits
