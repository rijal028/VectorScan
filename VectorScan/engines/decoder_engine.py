import numpy as np
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


# ================= CONFIG =================
K = 10
TOP_N = 100


# ================= EMBEDDING EXTRACTION =================
def extract_embeddings(model):
    return model.transformer.wte.weight.detach().cpu().numpy()


# ================= EMBEDDING DRIFT =================
def compute_embedding_drift(base_emb, ft_emb):
    sims = np.sum(base_emb * ft_emb, axis=1) / (
        np.linalg.norm(base_emb, axis=1) *
        np.linalg.norm(ft_emb, axis=1)
    )
    return 1 - sims


# ================= GEOMETRY =================
def get_top_k_neighbors(embedding_matrix, token_id, k):
    sims = cosine_similarity(
        embedding_matrix[token_id].reshape(1, -1),
        embedding_matrix
    )[0]

    sorted_indices = np.argsort(-sims)
    sorted_indices = sorted_indices[sorted_indices != token_id]
    return sorted_indices[:k]


def compute_geometry_top_n(base_emb, ft_emb, drift_scores):
    sorted_indices = np.argsort(-drift_scores)
    top_indices = sorted_indices[:TOP_N]

    stabilities = []

    for token_id in top_indices:
        base_n = set(get_top_k_neighbors(base_emb, token_id, K))
        ft_n = set(get_top_k_neighbors(ft_emb, token_id, K))
        overlap = len(base_n.intersection(ft_n))
        stabilities.append(overlap / K)

    return np.array(stabilities), top_indices


# ================= NEIGHBOR SHIFT =================
def compute_neighbor_shift(base_emb, ft_emb, token_id, tokenizer):
    base_vec = base_emb[token_id]
    ft_vec = ft_emb[token_id]

    base_sims = cosine_similarity(base_vec.reshape(1, -1), base_emb)[0]
    ft_sims = cosine_similarity(ft_vec.reshape(1, -1), ft_emb)[0]

    delta = ft_sims - base_sims
    delta[token_id] = 0.0

    closer_ids = np.argsort(-delta)[:10]
    farther_ids = np.argsort(delta)[:10]

    closer = [
        {"token": tokenizer.decode([int(i)]), "delta": float(delta[i])}
        for i in closer_ids
    ]

    farther = [
        {"token": tokenizer.decode([int(i)]), "delta": float(delta[i])}
        for i in farther_ids
    ]

    return closer, farther


# ================= MAIN ENGINE =================
def run_decoder_analysis(baseline_path, finetuned_path, probe_prompts):

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    model_base = AutoModelForCausalLM.from_pretrained(baseline_path)
    model_ft = AutoModelForCausalLM.from_pretrained(finetuned_path)

    print("Extracting embeddings...")
    base_emb = extract_embeddings(model_base)
    ft_emb = extract_embeddings(model_ft)

    print("Computing embedding drift...")
    drift_scores = compute_embedding_drift(base_emb, ft_emb)

    mean_drift = float(np.mean(drift_scores))
    top_1_percent = float(np.mean(np.sort(drift_scores)[-int(0.01 * len(drift_scores)):]))
    max_drift = float(np.max(drift_scores))

    print("Analyzing top drift tokens...")
    geometry_scores, top_indices = compute_geometry_top_n(
        base_emb, ft_emb, drift_scores
    )

    mean_stability = float(np.mean(geometry_scores))

    detailed_tokens = []

    for token_id in top_indices:
        closer, farther = compute_neighbor_shift(
            base_emb, ft_emb, token_id, tokenizer
        )

        detailed_tokens.append({
            "token": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "drift_score": float(drift_scores[token_id]),
            "neighbors_getting_closer": closer,
            "neighbors_getting_farther": farther
        })

    # ================= BEHAVIOR =================
    kl_scores = []
    cosine_scores = []
    l2_scores = []
    base_entropies = []
    ft_entropies = []

    print("Computing behavioral drift...")

    for prompt in probe_prompts:

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            out_base = model_base(**inputs)
            out_ft = model_ft(**inputs)

        logits_base = out_base.logits[0, -1, :]
        logits_ft = out_ft.logits[0, -1, :]

        prob_base = F.softmax(logits_base, dim=-1).cpu().numpy()
        prob_ft = F.softmax(logits_ft, dim=-1).cpu().numpy()

        kl_scores.append(entropy(prob_base, prob_ft))

        cosine_scores.append(
            F.cosine_similarity(
                logits_base.unsqueeze(0),
                logits_ft.unsqueeze(0)
            ).item()
        )

        l2_scores.append(torch.norm(logits_base - logits_ft).item())

        base_entropies.append(entropy(prob_base))
        ft_entropies.append(entropy(prob_ft))

    entropy_delta = np.array(base_entropies) - np.array(ft_entropies)

    report = {
        "embedding_layer": {
            "mean_drift": mean_drift,
            "top_1_percent_mean": top_1_percent,
            "max_drift": max_drift
        },
        "geometry_layer": {
            "top_n_analyzed": TOP_N,
            "mean_neighbor_stability_on_top_n": mean_stability
        },
        "behavior_layer": {
            "mean_kl_divergence": float(np.mean(kl_scores)),
            "max_kl_divergence": float(np.max(kl_scores))
        },
        "logit_layer": {
            "mean_cosine_similarity": float(np.mean(cosine_scores)),
            "mean_l2_distance": float(np.mean(l2_scores))
        },
        "entropy_layer": {
            "mean_base_entropy": float(np.mean(base_entropies)),
            "mean_ft_entropy": float(np.mean(ft_entropies)),
            "mean_entropy_delta": float(np.mean(entropy_delta))
        },
        "top_drift_tokens_analysis": detailed_tokens
    }

    return report