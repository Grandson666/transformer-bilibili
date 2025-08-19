# transformer-bilibili

## Project Overview

`transformer-bilibili` is a minimal and educational implementation of the Transformer model, designed for sequence-to-sequence tasks using PyTorch. The project demonstrates the core logic of Transformer architectures, including self-attention, multi-head attention, positional encoding, and training on simple copy tasks.

## Project Structure

```
transformer-bilibili/
│
├── data.py
├── model.py
├── train.py
├── utils.py
└── __pycache__/
```

## File Details

### data.py

- Defines special token indices (`PAD_TOKEN`, `SOS_TOKEN`, `EOS_TOKEN`).
- Contains the function `create_copy_data_batch` to generate random sequence data for model training, including source sequences, target input/output, and masks for both encoder and decoder.

### model.py

- Implements the core Transformer components:
  - `Embeddings` and `PositionalEncoding` for sequence input representation.
  - `MultiHeadAttention`, `FeedForward`, and `AddNorm` as building blocks.
  - `EncoderLayer` and `DecoderLayer` to stack multiple layers.
  - `Transformer` class tying together encoder and decoder logic, supporting encode, decode, and forward passes for modeling.

### train.py

- Provides functions for model training (`train`) and evaluation (`run_example`).
- Initializes hyperparameters, model, optimizer, and loss function.
- Runs the training loop over generated data batches and prints training progress.
- Evaluates the model by generating output from a test input sequence.

### utils.py

- Utility functions for mask creation:
  - `create_padding_mask`: Creates masks to ignore padding tokens in sequences.
  - `create_subsequent_mask`: Creates lower triangular masks for decoder self-attention, ensuring causal prediction.

## Quick Start

1. Install dependencies:  
   ```
   pip install torch
   ```

2. Run training and evaluation:  
   ```
   python train.py
   ```

## License

This repository is for educational use and does not include a specific license.
