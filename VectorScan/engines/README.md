# Engines Folder

This folder contains architecture-specific drift engines.

## encoder_engine.py
Used for encoder-based models (e.g., DistilBERT).

Focus:
- Embedding drift
- Geometry stability
- Masked token distribution analysis

## decoder_engine.py
Used for decoder-based models (e.g., GPT2).

Focus:
- Embedding drift
- Next-token probability redistribution
- Logit shift
- Entropy change

The main.py script automatically detects model architecture and routes to the appropriate engine.
