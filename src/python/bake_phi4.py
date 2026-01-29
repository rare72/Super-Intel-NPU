import os
import shutil
import argparse
import logging
import sys
import subprocess
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "microsoft/Phi-4-mini-flash-reasoning"
# PyTorch Model - No specific revision branch logic needed for simple download
REVISION_BRANCH = "main"
# Download everything (no specific pattern filtering needed for PyTorch typically, but good to avoid massive unused files if any)
ALLOW_PATTERNS = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("BakePhi4Flash")

def bake_phi4(staging_dir, output_dir):
    logger = setup_logging()

    # ENSURE DIRECTORIES EXIST (Windows Fix)
    # Resolve absolute paths to avoid confusion and ensure compatibility
    staging_dir = os.path.abspath(staging_dir)
    output_dir = os.path.abspath(output_dir)

    try:
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)

    logger.info(f"--- Phi-4-Mini-Flash-Reasoning Bake Process ---")
    logger.info(f"Target Model: {MODEL_ID}")
    logger.info(f"Staging Directory: {staging_dir}")
    logger.info(f"Output Directory:  {output_dir}")

    # 1. Download (Phase 1)
    logger.info(f">>> [Phase 1] Downloading Model to {staging_dir}...")

    # Check if download already exists (User might have manually downloaded)
    # If directory is not empty, assume download is present or partially present
    if os.path.exists(staging_dir) and os.listdir(staging_dir):
        logger.info(f"Directory {staging_dir} is not empty. Assuming files exist or resuming.")

    model_name_clean = MODEL_ID.split("/")[-1]
    # Use valid Windows path for cache if possible, or relative
    hf_cache_path = os.path.abspath(f"cache/model_{model_name_clean}")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=staging_dir,
            cache_dir=hf_cache_path,
            local_dir_use_symlinks=False
        )
        logger.info("[Success] Download complete.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # 2. Convert/Quantize (Phase 2)
    conversion_dir = os.path.abspath("model/model_staging")
    if os.path.exists(conversion_dir):
        shutil.rmtree(conversion_dir)
    os.makedirs(conversion_dir, exist_ok=True)

    logger.info(f">>> [Phase 2] Converting to OpenVINO IR at {conversion_dir}...")

    # PyTorch Source -> Optimum CLI
    # target_input_dir is just the staging root since it's a direct download
    target_input_dir = staging_dir

    logger.info("    > Detected PyTorch source. Using optimum-cli.")
    # INT8 Conversion requested
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", target_input_dir,
        "--task", "text-generation-with-past",
        "--weight-format", "int8", # INT8 QUANTIZATION
        "--trust-remote-code", # REQUIRED for Phi-4-mini-flash-reasoning (custom code)
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
        # Trust remote code for tokenizer as well
        tokenizer = AutoTokenizer.from_pretrained(target_input_dir, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        logger.info("[Success] Tokenizer verified/saved.")
    except:
        logger.warning("Could not auto-load tokenizer from input dir. Manually checking for files...")
        # Copy tokenizer files manually if they exist
        for f in os.listdir(target_input_dir):
            if "token" in f or "vocab" in f or "special_tokens" in f:
                shutil.copy2(os.path.join(target_input_dir, f), os.path.join(output_dir, f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging_dir", default="model/model_template", help="Directory for raw download")
    parser.add_argument("--output_dir", default="models/model_CURRENT", help="Directory for final output")
    args = parser.parse_args()

    bake_phi4(args.staging_dir, args.output_dir)
