# Models Folder

This folder stores:

- Baseline models (before fine-tuning)
- Fine-tuned models (after developer training)

VectorScan does NOT perform fine-tuning.

Developers must place:

models/baseline/<model_name>  
models/finetuned/<model_name>

VectorScan will compare both to detect representation drift.

Models are not included in this repository due to size constraints.