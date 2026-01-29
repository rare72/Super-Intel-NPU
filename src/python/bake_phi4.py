import os
import shutil
import argparse
import logging
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "AhtnaGlen/phi-4-mini-instruct-int4-sym-npu-ov"
REVISION_BRANCH = "main"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("BakePhi4PreBaked")

def bake_phi4(staging_dir, output_dir):
    logger = setup_logging()

    # ENSURE DIRECTORIES EXIST
    staging_dir = os.path.abspath(staging_dir)
    output_dir = os.path.abspath(output_dir)

    try:
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)

    logger.info(f"--- Phi-4 Bake Process (Pre-Baked OV) ---")
    logger.info(f"Target Model: {MODEL_ID}")
    logger.info(f"Staging Directory: {staging_dir}")
    logger.info(f"Output Directory:  {output_dir}")

    # 1. Download (Phase 1)
    logger.info(f">>> [Phase 1] Downloading OpenVINO Model to {staging_dir}...")

    model_name_clean = MODEL_ID.split("/")[-1]
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

    # 2. Skip Conversion (Phase 2 - Skipped)
    # The source is already a baked OpenVINO IR (XML/BIN)
    logger.info(f">>> [Phase 2] Skipping Conversion (Model is pre-baked OpenVINO IR)...")

    # 3. Publish (Phase 3)
    logger.info(f">>> [Phase 3] Publishing to Final Directory: {output_dir}...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Just copy the downloaded files to the output
    try:
        shutil.copytree(staging_dir, output_dir)
        logger.info("[Success] Model published.")
    except Exception as e:
        logger.error(f"Failed to publish model: {e}")
        sys.exit(1)

    # 4. Tokenizer Verification
    try:
        # Trust remote code just in case, though usually not needed for pre-baked if config is standard
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        logger.info("[Success] Tokenizer loaded and verified.")
    except Exception as e:
        logger.warning(f"Tokenizer warning: {e}. Ensure tokenizer.json/config.json exist in output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging_dir", default="model/model_template", help="Directory for raw download")
    parser.add_argument("--output_dir", default="models/model_CURRENT", help="Directory for final output")
    args = parser.parse_args()

    bake_phi4(args.staging_dir, args.output_dir)
