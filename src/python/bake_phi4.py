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

    logger.info(f"--- Phi-4 Bake Process ---")
    logger.info(f"Target Model: {MODEL_ID} (GPU Branch)")
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
            allow_patterns=ALLOW_PATTERNS,
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

    # Robust Input Directory Resolution (Windows recursive search)
    target_input_dir = staging_dir
    is_onnx_source = False

    # Walk to find the actual model files (handling gpu/ subfolders)
    for root, dirs, files in os.walk(staging_dir):
        # Look for typical model files
        if any(f.endswith(".onnx") for f in files) or "config.json" in files:
             # Heuristic: if 'gpu' is in path, it's likely the one we want
             if "gpu" in root.lower():
                target_input_dir = root
                if any(f.endswith(".onnx") for f in files):
                    is_onnx_source = True
                break

    logger.info(f"    > Resolved Input Directory: {target_input_dir}")

    # Toolchain Selection based on Source Type
    if is_onnx_source:
        logger.info("    > Detected ONNX source files. Switching to 'ovc' (OpenVINO Converter).")
        onnx_file = next((f for f in os.listdir(target_input_dir) if f.endswith(".onnx")), None)

        if not onnx_file:
            logger.error("No ONNX file found in target directory despite detection.")
            sys.exit(1)

        input_model_path = os.path.join(target_input_dir, onnx_file)

        # OVC Command for ONNX -> IR
        cmd = [
            "ovc",
            input_model_path,
            "--output_model", os.path.join(conversion_dir, "openvino_model.xml"),
            # Note: We do NOT pass --weight-format here as ovc handles compression differently
            # and source is likely already quantized (int4).
        ]
    else:
        logger.info("    > Detected PyTorch source. Using optimum-cli.")
        # Reverted to INT4 per user instruction to "IGNORE INT4/INT8 conversion"
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
