import os
import json
import argparse
import shutil
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
            new_shapes = {}
            for input_node in ov_model.inputs:
                partial_shape = input_node.get_partial_shape()
                # Default to [1, 1..4096] for 2+ dims (usually input_ids, attention_mask)
                if len(partial_shape) >= 2:
                    batch_dim = ov.Dimension(1)
                    seq_dim = ov.Dimension(1, 4096)

                    new_shape = [batch_dim, seq_dim]
                    if len(partial_shape) > 2:
                        for i in range(2, len(partial_shape)):
                            new_shape.append(partial_shape[i])

                    new_shapes[input_node.any_name] = ov.PartialShape(new_shape)
                    print(f"    > Setting shape for {input_node.any_name}: {new_shape}")

            if new_shapes:
                ov_model.reshape(new_shapes)
                ov.save_model(ov_model, xml_path)
                print(">>> [Bake] Remediation Applied. Verifying...")

                # VERIFICATION STEP
                # Reload to prove it worked
                verify_model = core.read_model(xml_path)
                success = True
                for input_node in verify_model.inputs:
                    ps = input_node.get_partial_shape()
                    print(f"    > [Verify] {input_node.any_name}: {ps}")
                    for i, dim in enumerate(ps):
                        if dim.is_dynamic and (dim.get_max_length() == -1 or dim.get_max_length() > 200000):
                            print(f"[ERROR] Node {input_node.any_name} Dimension {i} is still unbounded! NPU will crash.")
                            success = False

                if not success:
                    print(">>> [Bake] FATAL: Remediation failed to bound all shapes.")
                else:
                    print(">>> [Bake] Verification Passed: All shapes are bounded.")
            else:
                print("[Warning] No suitable inputs found to reshape.")
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

    # Ensure output dir is clean if it exists to prevent partial overwrites
    if os.path.exists(args.output_dir):
        print(f">>> [Bake] Warning: Output directory {args.output_dir} exists.")

    bake_model(args.model_id, args.staging_dir, args.output_dir, args.config)
