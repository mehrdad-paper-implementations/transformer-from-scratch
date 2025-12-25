import json
import torch
from types import SimpleNamespace
from src.transformer import Transformer


def generate_causal_mask(seq_len):
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    )  # [T, T]

    return ~mask


def main(config: SimpleNamespace):
    model = Transformer(n_heads=config.num_heads,
                        num_encoder_layers=config.num_layers,
                        num_decoder_layers=config.num_layers,
                        d_model=config.d_model,
                        d_ff=config.d_ff,
                        input_vocab_size=config.vocab_size,
                        output_vocab_size=config.vocab_size,
                        dropout=config.dropout,
                        d_emb=config.d_emb)

    x = torch.randint(low=0, high=100, size=(config.batch_size, 10))  # [B, T_enc]
    y = torch.randint(low=0, high=100, size=(config.batch_size, 12))  # [B, T_dec]

    causal_mask = generate_causal_mask(y.size(1))
    causal_mask = causal_mask.unsqueeze(0).expand(y.size(0), -1, -1)  # [B, T, T]

    logits = model(input_ids=x,
                   output_ids=y,
                   causal_mask=causal_mask)

    assert logits.shape == (config.batch_size, y.size(1), config.vocab_size)


if __name__ == '__main__':
    with open('./configs/transformer_config.json', 'r') as file:
        config = json.load(file)
        config = SimpleNamespace(**config)

    main(config)
