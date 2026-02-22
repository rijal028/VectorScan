# Experiments

This folder contains drift analysis results across:

- Encoder model (DistilBERT)
- Decoder model (GPT2)
- Epoch 1 vs Epoch 5 fine-tuning comparison

Purpose:
To demonstrate how VectorScan detects drift patterns across architectures and training intensity.

Observations:

- Encoder shows stronger embedding drift.
- Decoder shows stronger behavioral (probability) drift.
- Higher epochs increase KL divergence and entropy shifts.
