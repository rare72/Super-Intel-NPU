import os
import shutil
import argparse
import logging
import sys
import subprocess
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "microsoft/phi-4-onnx"
# Per instructions: Download GPU branch
REVISION_BRANCH = "main" # The repo structure seems to be folders, not branches, but we filter by allow_patterns
ALLOW_PATTERNS = ["gpu/*"]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("BakePhi4")

def bake_phi4(staging_dir, output_dir):
    logger = setup_logging()
    logger.info(f"--- Phi-4 Bake Process ---")
    logger.info(f"Target Model: {MODEL_ID} (GPU Branch)")

    # 1. Download (Phase 1)
    logger.info(f">>> [Phase 1] Downloading Model to {staging_dir}...")

    model_name_clean = MODEL_ID.split("/")[-1]
    hf_cache_path = f"/Super-Intel-NPU/cache/model_{model_name_clean}"

    try:
        # We download to staging_dir (model/model_template)
        # Note: The prompt instructions say: "download microsoft/phi-4-onnx --include gpu/*"
        # Since it's an ONNX repo, it likely has .onnx files.
        # However, Phase 2 instructions say: "Use optimum-cli to convert the Phi-4 model... --task text-generation-with-past"
        # Optimum-CLI usually takes a PyTorch model as input for export, OR it can optimize an existing ONNX.
        # IF the repo is already ONNX, optimum-cli export might not be the right command if we are just Quantizing.
        # BUT, the instructions explicitly said: "optimum-cli export openvino ... --model microsoft/phi-4-onnx"
        # This implies optimum-cli handles the ONNX->OpenVINO conversion or PyTorch->OpenVINO.
        # Given "microsoft/phi-4-onnx" is the ID, it might be pre-exported.
        # BUT, the instructions usually imply starting from a base or following a specific NPU conversion path.
        # Let's follow the instruction: Download first, then run optimum-cli pointing to it.

        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=staging_dir,
            cache_dir=hf_cache_path,
            allow_patterns=ALLOW_PATTERNS,
            local_dir_use_symlinks=False
        )
        logger.info("[Success] Download complete.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # 2. Convert/Quantize (Phase 2)
    # Output to model/model_staging
    conversion_dir = "model/model_staging"
    if os.path.exists(conversion_dir):
        shutil.rmtree(conversion_dir)
    os.makedirs(conversion_dir, exist_ok=True)

    logger.info(f">>> [Phase 2] Converting to OpenVINO IR (INT4) at {conversion_dir}...")

    # Construct optimum-cli command
    # optimum-cli export openvino --model <INPUT> --task text-generation-with-past --weight-format int4 --sym --ratio 1.0 --group-size 128 <OUTPUT>

    # Input is the downloaded directory. However, because we downloaded 'gpu/*' subfolder, the actual model might be in staging_dir/gpu/gpu-int4-rtn-block-32 etc.
    # We need to find the specific folder containing the model file.
    # For now, let's point to staging_dir and assume optimum figures it out or we point to the GPU subfolder.
    # If the user said "Download gpu/*", and we used local_dir=staging_dir, then the files are in staging_dir/gpu/...

    # Let's search for the folder containing .onnx files or config.json
    target_input_dir = staging_dir
    for root, dirs, files in os.walk(staging_dir):
        if any(f.endswith(".onnx") for f in files) or "config.json" in files:
            # Heuristic: Prefer the deepest folder with onnx files or config
            # specifically looking for the gpu one
            if "gpu" in root:
                target_input_dir = root
                break

    logger.info(f"    > Resolved Input Directory: {target_input_dir}")

    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", target_input_dir,
        "--task", "text-generation-with-past",
        "--weight-format", "int4",
        "--sym",
        "--ratio", "1.0",
        "--group-size", "128",
        conversion_dir
    ]

    logger.info(f"    > Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("[Success] Conversion complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

    # 3. Publish (Phase 3)
    logger.info(f">>> [Phase 3] Publishing to Final Directory: {output_dir}...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    shutil.copytree(conversion_dir, output_dir)
    logger.info("[Success] Model published.")

    # 4. Tokenizer Handling
    # Ensure tokenizer is present. If optimum-cli didn't copy it (sometimes it doesn't if input was ONNX), copy from source.
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_input_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("[Success] Tokenizer verified/saved.")
    except:
        logger.warning("Could not auto-load tokenizer from input dir. Manually checking for files...")
        # Copy tokenizer files manually if they exist
        for f in os.listdir(target_input_dir):
            if "token" in f or "vocab" in f:
                shutil.copy2(os.path.join(target_input_dir, f), os.path.join(output_dir, f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging_dir", default="model/model_template", help="Directory for raw download")
    parser.add_argument("--output_dir", default="models/model_CURRENT", help="Directory for final output")
    args = parser.parse_args()

    bake_phi4(args.staging_dir, args.output_dir)
