import os
import json
from transformers import AutoConfig
from engines.decoder_engine import run_decoder_analysis
from engines.encoder_engine import run_encoder_analysis


BASELINE_DIR = "models/baseline"
FINETUNED_DIR = "models/finetuned"

PROBE_PROMPTS = [
    "Cyber attacks exploit system",
    "The organization improves network",
    "Sensitive data requires strong",
    "Hackers attempt to bypass",
    "Security teams analyze potential"
]


def get_first_model_path(directory):
    models = os.listdir(directory)
    if not models:
        raise ValueError(f"No model found in {directory}")
    return os.path.join(directory, models[0])


def detect_model_architecture(model_path):

    config = AutoConfig.from_pretrained(model_path)

    model_type = config.model_type

    # Decoder-only models
    decoder_types = [
        "gpt2",
        "gpt_neo",
        "gptj",
        "gpt_neox",
        "llama",
        "mistral",
        "falcon"
    ]

    # Encoder-only models
    encoder_types = [
        "bert",
        "roberta",
        "distilbert",
        "albert",
        "electra",
        "deberta"
    ]

    if model_type in decoder_types:
        return "decoder"

    if model_type in encoder_types:
        return "encoder"

    # Fallback logic
    if hasattr(config, "is_decoder") and config.is_decoder:
        return "decoder"

    raise ValueError(f"Unsupported model type: {model_type}")

def main():

    baseline_path = get_first_model_path(BASELINE_DIR)
    finetuned_path = get_first_model_path(FINETUNED_DIR)

    print("Baseline:", baseline_path)
    print("Finetuned:", finetuned_path)

    model_type = detect_model_architecture(baseline_path)

    print("Detected model type:", model_type)

    if model_type == "decoder":
        report = run_decoder_analysis(
            baseline_path,
            finetuned_path,
            PROBE_PROMPTS
        )

    elif model_type == "encoder":
        report = run_encoder_analysis(
            baseline_path,
            finetuned_path,
            PROBE_PROMPTS
        )

    else:
        raise ValueError("Unknown model type")

    os.makedirs("reports", exist_ok=True)

    with open("reports/vectorscan_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Report saved.")


if __name__ == "__main__":
    main()