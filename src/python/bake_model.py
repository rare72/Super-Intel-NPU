import os
import json
import argparse
import shutil
import gc
import logging
import traceback
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import openvino as ov

# --- CONSTANTS ---
STATIC_SEQ_LEN = 1024
STATIC_BATCH_SIZE = 1

def setup_logging(verbose=False, log_file="bake_model.log"):
    """
    Configures logging to file and stdout.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file, mode='w')

    c_handler.setLevel(level)
    f_handler.setLevel(logging.DEBUG) # Always capture debug in file

    # Create formatters and add it to handlers
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    c_format = logging.Formatter(format_str)
    f_format = logging.Formatter(format_str)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Set root to debug, handlers filter
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def bake_model(model_id, staging_dir, output_dir, config_path):
    """
    Downloads, Converts to IR, Applies NNCF INT4 Compression, and Reshapes for NPU.
    """
    logger = logging.getLogger()
    logger.info(f">>> [Bake] Starting process for model: {model_id}")

    try:
        # 1. Download Open-Weight Model
        logger.info(f">>> [Bake] Downloading to {staging_dir}...")
        snapshot_download(repo_id=model_id, local_dir=staging_dir)

        # 2. Load NNCF Configuration
        logger.info(f">>> [Bake] Loading NNCF config from {config_path}...")
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        quantization_config = {
            "bits": config_data.get("bits", 4),
            "sym": config_data.get("sym", False),
            "group_size": config_data.get("group_size", 128),
            "ratio": config_data.get("ratio", 1.0),
        }

        # 3. Load & Compress
        logger.info(">>> [Bake] Loading model and applying Data-Free NNCF INT4 Compression...")
        model = OVModelForCausalLM.from_pretrained(
            staging_dir,
            export=True,
            compile=False,
            load_in_8bit=False,
            quantization_config=quantization_config
        )

        # 3b. IN-MEMORY RESHAPE (Fix for Bus Error)
        # Instead of saving then re-reading (which causes mmap conflicts),
        # we reshape the graph object directly in memory before the first save.
        logger.info(f">>> [Bake] Remediation: Enforcing STRICT STATIC shapes [{STATIC_BATCH_SIZE}, {STATIC_SEQ_LEN}] in-memory...")

        # Access the underlying OpenVINO model object
        ov_model_obj = model.model
        new_shapes = {}

        for input_node in ov_model_obj.inputs:
            name = input_node.any_name
            partial_shape = input_node.get_partial_shape()

            # 1. Handle beam_idx (Strictly 1D: [Batch])
            if "beam_idx" in name:
                new_shapes[name] = ov.PartialShape([STATIC_BATCH_SIZE])
                logger.info(f"    > Locking {name} to {[STATIC_BATCH_SIZE]}")
                continue

            # 2. Handle Primary Inputs (input_ids, attention_mask, position_ids) -> [Batch, SeqLen]
            if any(k in name for k in ["input_ids", "attention_mask", "position_ids"]):
                if len(partial_shape) >= 2:
                    new_shapes[name] = ov.PartialShape([STATIC_BATCH_SIZE, STATIC_SEQ_LEN])
                    logger.info(f"    > Locking {name} to {[STATIC_BATCH_SIZE, STATIC_SEQ_LEN]}")
                continue

            # 3. Handle Other Inputs (e.g. past_key_values) - Log only for now
            if partial_shape.is_dynamic:
                logger.warning(f"    > [Warning] Found dynamic input '{name}' with shape {partial_shape}. Not reshaping automatically.")

        if new_shapes:
            logger.info("Applying reshape directly to in-memory graph...")
            # This updates the model object held by 'model' (OVModelForCausalLM)
            # FIX: Call reshape on the underlying openvino.runtime.Model object (ov_model_obj)
            # because OVModelForCausalLM.reshape() expects specific args (batch, seq_len)
            # and does not accept a dictionary of PartialShapes.
            ov_model_obj.reshape(new_shapes)
        else:
            logger.warning("[Warning] No suitable inputs found to reshape.")

        # 4. Save Optimized & Reshaped Binary
        logger.info(f">>> [Bake] Saving optimized binaries to {output_dir}...")
        model.save_pretrained(output_dir)

        # 5. Save Tokenizer
        logger.info(">>> [Bake] Saving Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(staging_dir)
        tokenizer.save_pretrained(output_dir)

        # CRITICAL MEMORY CLEANUP
        logger.info(">>> [Bake] Cleaning up memory...")
        del model
        del ov_model_obj
        gc.collect()

        # 6. Verification (Safe Read-Only Check)
        logger.info(">>> [Bake] Verifying static shapes...")
        core = ov.Core()
        xml_path = os.path.join(output_dir, "openvino_model.xml")

        if os.path.exists(xml_path):
            verify_model = core.read_model(xml_path)
            success = True
            for input_node in verify_model.inputs:
                ps = input_node.get_partial_shape()
                logger.info(f"    > [Verify] {input_node.any_name}: {ps}")
                if ps.is_dynamic:
                    logger.error(f"[ERROR] Node {input_node.any_name} is still dynamic! NPU requires STATIC shapes.")
                    success = False

            if not success:
                logger.critical(">>> [Bake] FATAL: Failed to make model fully static.")
                sys.exit(1)
            else:
                logger.info(">>> [Bake] Verification Passed: Model is fully static.")
        else:
             logger.error(">>> [Bake] Error: Saved model file not found.")

        logger.info(f">>> [Bake] Success! Optimized NPU-ready model is at {output_dir}")

    except Exception as e:
        logger.error(f"Bake Process Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake a model into NNCF INT4 OpenVINO format.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace Model ID")
    parser.add_argument("--staging_dir", type=str, default="./model_staging", help="Directory for raw download")
    parser.add_argument("--output_dir", type=str, default="./offering_int4_binary", help="Directory for final output")
    parser.add_argument("--config", type=str, default="src/python/nncf_config.json", help="Path to NNCF config json")

    # Logging Args
    parser.add_argument("--verbose", action="store_true", help="Enable verbose stdout logging")
    parser.add_argument("--log_file", type=str, default="bake_model.log", help="Path to log file")

    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    if os.path.exists(args.output_dir):
        logging.warning(f">>> [Bake] Warning: Output directory {args.output_dir} exists.")

    bake_model(args.model_id, args.staging_dir, args.output_dir, args.config)
