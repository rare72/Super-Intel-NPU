import os
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import openvino as ov

def bake_model(model_id, staging_dir, output_dir, config_path):
    """
    Downloads, Converts to IR, Applies NNCF INT4 Compression, and Reshapes for NPU.
    """
    print(f">>> [Bake] Starting process for model: {model_id}")

    # 1. Download Open-Weight Model
    print(f">>> [Bake] Downloading to {staging_dir}...")
    snapshot_download(repo_id=model_id, local_dir=staging_dir)

    # 2. Load NNCF Configuration
    print(f">>> [Bake] Loading NNCF config from {config_path}...")
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    quantization_config = {
        "bits": config_data.get("bits", 4),
        "sym": config_data.get("sym", False),
        "group_size": config_data.get("group_size", 128),
        "ratio": config_data.get("ratio", 1.0),
    }

    # 3. Load & Compress
    print(">>> [Bake] Loading model and applying Data-Free NNCF INT4 Compression...")
    model = OVModelForCausalLM.from_pretrained(
        staging_dir,
        export=True,
        compile=False,
        load_in_8bit=False,
        quantization_config=quantization_config
    )

    # 4. Save Initial Binary
    print(f">>> [Bake] Saving optimized binaries to {output_dir}...")
    model.save_pretrained(output_dir)

    # 5. NPU Shape Remediation (Post-Processing)
    print(">>> [Bake] Remediation: Reshaping model for NPU compatibility...")
    try:
        core = ov.Core()
        xml_path = os.path.join(output_dir, "openvino_model.xml")
        bin_path = os.path.join(output_dir, "openvino_model.bin")

        if os.path.exists(xml_path):
            ov_model = core.read_model(xml_path)

            # Inspect and Apply Reshape
            # NPU (Level Zero) often fails with unbounded dynamic shapes (e.g., -1).
            # We enforce Batch Size = 1 and Sequence Length = 1..4096 (Bounded Dynamic) or Static.
            # Using partial shape with bounds is the most robust method for 2025/2026 drivers.

            new_shapes = {}
            for input_node in ov_model.inputs:
                # Assuming standard inputs: input_ids, attention_mask, position_ids
                # Shape is usually [batch, seq_len]
                # We set [1, 1..4096]
                partial_shape = input_node.get_partial_shape()
                if partial_shape.rank.is_static and len(partial_shape) >= 2:
                    # Create bounded dimension for sequence length
                    # Dimension(min, max)
                    batch_dim = ov.Dimension(1)
                    seq_dim = ov.Dimension(1, 4096)

                    new_shape = [batch_dim, seq_dim]
                    # Preserve other dimensions if any (e.g. past_key_values if present in inputs)
                    if len(partial_shape) > 2:
                        for i in range(2, len(partial_shape)):
                            new_shape.append(partial_shape[i])

                    new_shapes[input_node.any_name] = ov.PartialShape(new_shape)
                    print(f"    > Reshaping {input_node.any_name} to {new_shape}")

            ov_model.reshape(new_shapes)

            # Overwrite the model
            ov.save_model(ov_model, xml_path)
            print(">>> [Bake] Remediation Complete: Model saved with bounded shapes.")
        else:
            print("[Warning] XML file not found for reshaping.")

    except Exception as e:
        print(f"[Warning] Shape remediation failed: {e}. NPU might be unstable.")

    # 6. Save Tokenizer
    print(">>> [Bake] Saving Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(staging_dir)
    tokenizer.save_pretrained(output_dir)

    print(f">>> [Bake] Success! Optimized NPU-ready model is at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake a model into NNCF INT4 OpenVINO format.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace Model ID")
    parser.add_argument("--staging_dir", type=str, default="./model_staging", help="Directory for raw download")
    parser.add_argument("--output_dir", type=str, default="./offering_int4_binary", help="Directory for final output")
    parser.add_argument("--config", type=str, default="src/python/nncf_config.json", help="Path to NNCF config json")

    args = parser.parse_args()

    bake_model(args.model_id, args.staging_dir, args.output_dir, args.config)
