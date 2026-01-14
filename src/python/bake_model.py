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
STATIC_SEQ_LEN = 128
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
    Uses a 2-Stage Process:
    1. Optimum Export + NNCF -> Intermediate IR (Dynamic)
    2. OpenVINO Core Load -> Reshape -> Final IR (Static)
    """
    logger = logging.getLogger()
    logger.info(f">>> [Bake] Starting process for model: {model_id}")

    intermediate_dir = os.path.join(output_dir, "intermediate_dynamic")

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

        # 3. STAGE 1: Load & Compress (Dynamic Shapes allowed here)
        logger.info(">>> [Bake] Stage 1: Exporting + Compressing (Dynamic)...")
        # We explicitly disable cache to remove past_key_values
        model = OVModelForCausalLM.from_pretrained(
            staging_dir,
            export=True,
            compile=False,
            load_in_8bit=False,
            quantization_config=quantization_config,
            use_cache=False
        )

        # Save Intermediate
        logger.info(f">>> [Bake] Saving Intermediate (Dynamic) IR to {intermediate_dir}...")
        model.save_pretrained(intermediate_dir)

        # Save Tokenizer to final output as well
        tokenizer = AutoTokenizer.from_pretrained(staging_dir)
        tokenizer.save_pretrained(output_dir)

        # Cleanup Stage 1
        del model
        gc.collect()

        # 4. STAGE 2: Reshape to Strict Static (Pure OpenVINO)
        logger.info(">>> [Bake] Stage 2: Loading Intermediate for Static Reshaping...")
        core = ov.Core()
        xml_path = os.path.join(intermediate_dir, "openvino_model.xml")
        ov_model = core.read_model(xml_path)

        logger.info(f">>> [Bake] Remediation: Enforcing STRICT STATIC shapes [{STATIC_BATCH_SIZE}, {STATIC_SEQ_LEN}]...")

        new_shapes = {}
        for input_node in ov_model.inputs:
            name = input_node.any_name
            partial_shape = input_node.get_partial_shape()
            logger.info(f"    > Found Input: {name}, Shape: {partial_shape}")

            # 1. Handle beam_idx (if present)
            if "beam_idx" in name:
                new_shapes[name] = ov.PartialShape([STATIC_BATCH_SIZE])
                logger.info(f"      -> Locking {name} to {[STATIC_BATCH_SIZE]}")
                continue

            # 2. Handle 2D Inputs (input_ids, attention_mask, position_ids)
            if len(partial_shape) == 2:
                new_shapes[name] = ov.PartialShape([STATIC_BATCH_SIZE, STATIC_SEQ_LEN])
                logger.info(f"      -> Locking 2D Input {name} to {[STATIC_BATCH_SIZE, STATIC_SEQ_LEN]}")
                continue

            # 3. Handle KV Cache (Should be absent due to use_cache=False, but safety check)
            if len(partial_shape) == 4:
                 logger.warning(f"      -> Warning: Found 4D input {name} despite use_cache=False. Locking it.")
                 new_shape_list = []
                 for idx, dim in enumerate(partial_shape):
                    if idx == 0: new_shape_list.append(STATIC_BATCH_SIZE)
                    elif idx == 2 or idx == 3: # Heuristic for SeqLen dim
                        if dim.is_dynamic: new_shape_list.append(STATIC_SEQ_LEN)
                        else: new_shape_list.append(dim.get_length())
                    else:
                        new_shape_list.append(dim.get_length() if dim.is_static else 1) # Fallback
                 new_shapes[name] = ov.PartialShape(new_shape_list)

        if new_shapes:
            logger.info("Applying reshape to graph...")
            ov_model.reshape(new_shapes)
            # Propagate to freeze constants
            ov_model.validate_nodes_and_infer_types()

        # 5. Save Final Static Model
        final_xml_path = os.path.join(output_dir, "openvino_model.xml")
        final_bin_path = os.path.join(output_dir, "openvino_model.bin")

        logger.info(f">>> [Bake] Serializing Final Static IR to {output_dir}...")
        ov.save_model(ov_model, final_xml_path)

        # Cleanup Intermediate
        logger.info(">>> [Bake] Removing intermediate files...")
        shutil.rmtree(intermediate_dir, ignore_errors=True)

        # 6. Verification
        logger.info(">>> [Bake] Final Verification...")
        verify_model = core.read_model(final_xml_path)
        success = True

        for input_node in verify_model.inputs:
            ps = input_node.get_partial_shape()
            logger.info(f"    > [Verify Input] {input_node.any_name}: {ps}")
            if ps.is_dynamic:
                logger.error(f"[ERROR] Node {input_node.any_name} is still dynamic!")
                success = False

        if not success:
            logger.critical(">>> [Bake] FATAL: Failed to make model fully static.")
            sys.exit(1)

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
         # If it's a directory, warn.
         logging.warning(f">>> [Bake] Warning: Output directory {args.output_dir} exists. Overwriting...")

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    bake_model(args.model_id, args.staging_dir, args.output_dir, args.config)
