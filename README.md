# Transformer Decoder from Scratch

A small GPT-style, decoder-only transformer written from the ground up in PyTorch. It trains on the Bible as a character-level corpus and then writes its own (somewhat biblical) text.

I wanted to really understand what happens inside a language model, so I implemented the pieces myself instead of importing a ready-made transformer.

## What's inside

- **Character-level tokenizer**: every unique character in the text becomes a token
- **Masked self-attention heads**: scaled dot-product attention with a causal (lower-triangular) mask, so the model can only look backwards
- **Multi-head attention**: several heads run in parallel and are projected back together
- **Feed-forward layers** with ReLU and dropout
- **Transformer blocks** with pre-LayerNorm and residual connections
- **Token + positional embeddings**
- **Training loop** with AdamW and periodic train/validation loss estimates
- **Autoregressive generation** that samples one character at a time

## Default hyperparameters

| Setting | Value |
|---|---|
| Context length (block size) | 60 characters |
| Embedding size | 50 |
| Transformer layers | 4 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Dropout | 0.2 |
| Training steps | 10,000 |

The model is deliberately small so it trains on a laptop. It uses a GPU automatically if one is available.

## Run it

```bash
pip install torch
python transformer.py
```

The script reads `bible.txt`, prints the train/validation loss as it trains, and finishes by generating 500 new characters.

> Note: the script opens `transformer/bible.txt`. Either run it from the parent folder or change the path to `bible.txt`.

## Built with

Python · PyTorch

## Credits

The architecture follows the "Attention Is All You Need" paper, and Andrej Karpathy's "Let's build GPT" walkthrough was a big inspiration.
