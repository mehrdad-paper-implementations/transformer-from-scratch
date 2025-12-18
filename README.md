# Transformer (Paper-Accurate, Modular Implementation)

This repository contains a **from-scratch, modular implementation of the Transformer model** as introduced in *["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)*, implemented using **PyTorch**.

The goal of this repository is **educational and architectural clarity**, not end-to-end training or benchmarking. The model is implemented closely following the original paper, with clean separation of components and minimal abstractions.

---

## Project Scope

- Faithful implementation of the original Transformer architecture
- Fully **modular design** (each conceptual block lives in its own file)
- Uses **absolute positional encoding** as described in the paper
- Includes a minimal **sanity check** to verify forward pass correctness

### Out of Scope (by design)

- Training loops
- Dataset handling
- Optimizers, schedulers
- Loss functions
- Evaluation scripts

This repository focuses **only on the model architecture itself**.

---

## Repository Structure

```
C:.
|   LICENSE
|   README.md
|   simple_test.py
|
+---configs
|       transformer_config.json
|
\---src
    |   multi_head_attention.py
    |   positional_encoding.py
    |   scaled_dot_product_attention.py
    |   transformer.py
    |   transformer_decoder_block.py
    |   transformer_decoder_layer.py
    |   transformer_encoder_block.py
    |   transformer_encoder_layer.py
    |   __init__.py
```

---

## Component Overview

> **Naming convention note**  
> In this implementation, a *layer* corresponds to a single encoder/decoder layer (as in PyTorch), while *blocks* are responsible for stacking multiple layers.

### Core Attention

- **`scaled_dot_product_attention.py`**  
  Scaled dot-product attention with optional masking.

- **`multi_head_attention.py`**  
  Multi-head attention built on top of scaled dot-product attention.

### Positional Encoding

- **`positional_encoding.py`**  
  Absolute sinusoidal positional encoding as described in the original paper.

### Encoder

- **`transformer_encoder_layer.py`**  
  A single Transformer encoder layer (self-attention + feed-forward).

- **`transformer_encoder_block.py`**  
  Stacks multiple encoder layers to form the full encoder.

### Decoder

- **`transformer_decoder_layer.py`**  
  A single Transformer decoder layer (masked self-attention, cross-attention, feed-forward).

- **`transformer_decoder_block.py`**  
  Stacks multiple decoder layers to form the full decoder.

### Full Model

- **`transformer.py`**  
  Assembles the encoder and decoder into a complete Transformer model.

---

## Configuration

Model hyperparameters are defined via:

```
configs/transformer_config.json
```

This keeps architectural choices explicit and avoids hardcoding.

---

## Sanity Check

A minimal forward-pass test is provided:

```
simple_test.py
```

This script:

- Instantiates the Transformer
- Passes random tensors through the model
- Verifies shape consistency and basic correctness

No training or loss computation is performed.

---

## Design Principles

- Paper-accurate implementation
- PyTorch-style layer semantics
- Modular, readable code
- No training-specific concerns

---

## Intended Use

This repository is intended for studying and understanding the Transformer architecture and its internal components. It is not designed as a production-ready or optimized implementation.

---

## Reference

> Vaswani et al., *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*, 2017

---

## License

This project is licensed under the terms of the LICENSE file included in the repository.

