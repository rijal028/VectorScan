# VectorScan v2

**A Developer-Centric Framework for Detecting Representation Drift Across Encoder and Decoder Models**

Author: Rijal Saepuloh  
Independent Researcher  

---

## 🔍 What Is VectorScan?

VectorScan is a developer-side diagnostic engine that measures **how a model changes after fine-tuning**.

It does not evaluate accuracy.  
It does not classify domains.  
It measures **representation drift**.

VectorScan answers:

- How far did tokens move?
- Did conceptual neighborhoods change?
- Did probability behavior shift?
- Did model confidence increase?
- Did new directional bias emerge?

---

# 🧠 From VectorScan v1 to v2 (Major Upgrade)

## 🔹 VectorScan v1

- Compared baseline and fine-tuned embedding tables
- Reported nearest neighbors of drifted tokens
- Focused only on token-level embedding changes

### Limitations

- No behavioral analysis
- No probability distribution tracking
- No decoder compatibility
- No entropy or logit-level diagnostics
- No architecture adaptation

---

## 🔥 VectorScan v2

VectorScan v2 introduces a **multi-layer drift analysis framework**:

| Layer | What It Measures |
|-------|------------------|
| Embedding Drift | How far tokens moved (cosine-based drift) |
| Geometry Drift | Whether neighborhood structure changed |
| Behavioral Drift | KL divergence between output distributions |
| Logit Drift | Cosine & L2 distance between raw logits |
| Entropy Drift | Change in model confidence |

### New Capabilities

- Works for **Encoder models** (e.g., DistilBERT)
- Works for **Decoder LLMs** (e.g., GPT2 / DistilGPT2)
- Automatically adapts engine based on model architecture
- Uses Top-N drifted tokens for efficient analysis
- Designed for local machine constraints

---

# 🏗 Architecture-Aware Engine

VectorScan detects whether a model is:

- Encoder (Masked LM)
- Decoder (Causal LM)

And automatically routes to the correct analysis pipeline.

### Encoder Emphasis
- Embedding cluster tightening
- Representation specificity
- Semantic clustering behavior

### Decoder Emphasis
- Probability redistribution
- Confidence shifts
- Token likelihood reweighting

---

# 📊 Metrics Explained (Simple Version)

## 1️⃣ Embedding Drift
Measures how far tokens moved in vector space.

Large drift → conceptual shift.

---

## 2️⃣ Geometry Drift
Checks whether nearest neighbors changed.

If neighbors change → token identity may have shifted.

---

## 3️⃣ Behavioral Drift (KL Divergence)
Measures change in output probability distribution.

High KL → model answers differently.

---

## 4️⃣ Logit Drift
Measures change in raw output scores before softmax.

Even if embeddings barely move, logits can shift heavily.

---

## 5️⃣ Entropy Drift
Measures change in model confidence.

Lower entropy → sharper probability peaks → higher certainty.

---

# 🧪 Experimental Setup

Tested on:

- GPT2 (epoch 1 vs epoch 5)
- DistilBERT (epoch 1 vs epoch 5)
- 300-sample cybersecurity dataset

### Observations

- Encoder drift stronger at embedding layer
- Decoder drift stronger at probability layer
- Higher epochs increase behavioral divergence
- Drift can occur even if embedding shift is small

---

## Validation Summary

VectorScan v2 was validated on:

- GPT2 (epoch 1 vs 5)
- DistilBERT (epoch 1 vs 5)
- 300-sample cybersecurity dataset

Results confirm:

- Architecture-specific drift patterns
- Probability redistribution without major embedding shift
- Entropy confidence compression in decoder models

---

# ⚡ Performance Optimization

VectorScan v2:

- Analyzes only Top 100 most drifted tokens
- Avoids brute-force token comparisons
- Runs under 10 minutes on 16GB RAM CPU machine

Designed for sprint-level experimentation.

---

# 🎯 Use Cases

- Detect bias amplification after fine-tuning
- Identify probability manipulation
- Monitor LoRA directional shifts
- Detect emergent domain convergence
- Pre-deployment safety diagnostics

---

# 🔒 What VectorScan Is NOT

- Not a classifier
- Not a regulator
- Not an accuracy evaluator
- Not a dataset inspector

It is a **diagnostic drift microscope**.

---

# 🧩 Relationship to CBVP v2

VectorScan → detects what changed  
CBVP → classifies what domain it belongs to  

Together they form:

**Developer Safety Layer + Regulatory Layer**

---

# 📁 Folder Structure
VectorScan/ │ 

├── dataset/ │

    ├── cyber.txt │

├── models/ │  

    ├── baseline/ │
    
    └── finetuned/ │

├── engines/ │

    ├── encoder_engine.py │
    
    └── decoder_engine.py │

├── reports.txt │

    ├── vectorscan_report │

├── main.py 

└── README.md

Fine-tuning is developer responsibility.  
VectorScan analyzes resulting models only.

---

# 🏁 Conclusion

VectorScan v2 transforms the original token comparison method into a:

- Multi-metric
- Architecture-aware
- Efficient
- Developer-oriented
- Drift-focused diagnostic framework

It does not ask:

"Is the model accurate?"

It asks:

"How did the model change?"

---


VectorScan v2 is designed to make model evolution observable, measurable, and diagnosable.
