import os
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def bake_model(model_id, staging_dir, output_dir, config_path):
    """
    Downloads, Converts to IR, and Applies NNCF INT4 Compression.
    """
    print(f">>> [Bake] Starting process for model: {model_id}")

    # 1. Download Open-Weight Model
    print(f">>> [Bake] Downloading to {staging_dir}...")
    snapshot_download(repo_id=model_id, local_dir=staging_dir)

    # 2. Load NNCF Configuration
    print(f">>> [Bake] Loading NNCF config from {config_path}...")
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Extract quantization parameters from the loaded JSON
    # Note: optimum-intel expects specific keys for quantization_config
    quantization_config = {
        "bits": config_data.get("bits", 4),
        "sym": config_data.get("sym", False),
        "group_size": config_data.get("group_size", 128),
        "ratio": config_data.get("ratio", 1.0),
        # Advanced parameters might need to be passed differently depending on version
        # For simplicity and robustness with standard optimum-intel, we use the main keys
    }

    # 3. Load & Compress
    print(">>> [Bake] Loading model and applying Data-Free NNCF INT4 Compression...")
    # The 'quantization_config' triggers NNCF internally in optimum-intel
    model = OVModelForCausalLM.from_pretrained(
        staging_dir,
        export=True,
        compile=False,
        load_in_8bit=False, # Ensure we are targeting INT4 via config
        quantization_config=quantization_config
    )

    # 4. Save Final Binary
    print(f">>> [Bake] Saving optimized binaries to {output_dir}...")
    model.save_pretrained(output_dir)

    # 5. Save Tokenizer
    print(">>> [Bake] Saving Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(staging_dir)
    tokenizer.save_pretrained(output_dir)

    print(f">>> [Bake] Success! optimized model is ready at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake a model into NNCF INT4 OpenVINO format.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace Model ID")
    parser.add_argument("--staging_dir", type=str, default="./model_staging", help="Directory for raw download")
    parser.add_argument("--output_dir", type=str, default="./offering_int4_binary", help="Directory for final output")
    parser.add_argument("--config", type=str, default="src/python/nncf_config.json", help="Path to NNCF config json")

    args = parser.parse_args()

    bake_model(args.model_id, args.staging_dir, args.output_dir, args.config)
