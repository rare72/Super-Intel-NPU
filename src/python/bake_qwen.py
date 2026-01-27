import os
import json
import argparse
import shutil
import gc
import logging
import traceback
import sys
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
import openvino as ov

# --- CONSTANTS ---
STATIC_SEQ_LEN = 4096
STATIC_BATCH_SIZE = 1

def setup_logging(verbose=False, log_file="bake_qwen.log"):
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # stdout handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(level)
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # file handler
    f_handler = logging.FileHandler(log_file, mode='w')
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

def bake_qwen(staging_dir, output_dir, config_path):
    logger = logging.getLogger()
    logger.info(f">>> [Bake] Starting Qwen3 Process...")
    logger.info(f"    > Seq Len: {STATIC_SEQ_LEN}")
    logger.info(f"    > Staging: {staging_dir}")

    intermediate_dir = os.path.join(output_dir, "intermediate_dynamic")

    try:
        # 0. Pre-Bake Fix: Patch Qwen3 to Qwen2 if needed
        # OpenVINO 2025.4 supports Qwen2. Qwen3 architecture might be unknown.
        # We check config.json and force 'qwen2' if 'qwen3' is present.
        config_file = os.path.join(staging_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                cfg = json.load(f)

            if cfg.get("model_type") == "qwen3":
                logger.warning(">>> [Patch] Found 'qwen3' model_type. Patching to 'qwen2' for OpenVINO compatibility...")
                cfg["model_type"] = "qwen2"
                with open(config_file, 'w') as f:
                    json.dump(cfg, f, indent=2)

        # 1. Load NNCF Config
        with open(config_path, 'r') as f:
            nncf_cfg_data = json.load(f)

        quantization_config = {
            "bits": 4,
            "sym": False,
            "group_size": 128,
            "ratio": 1.0, # Full compression
            # Qwen specific: ignore norm layers if sensitive
            "ignored_scope": {"types": []}
        }

        # Cleanup Stale Cache
        if os.path.exists("./model_cache_qwen"):
            logger.info(">>> [Bake] Cleaning up stale NPU cache...")
            shutil.rmtree("./model_cache_qwen")

        # 2. Stage 1: Export & Compress (Dynamic)
        logger.info(">>> [Bake] Stage 1: Exporting + NNCF INT4...")
        sys.stdout.flush()

        # Use trust_remote_code=False for model to force internal Qwen2 implementation (More robust tracing)
        # Use attn_implementation="eager" to fix graph tracing issues with SDPA
        model = OVModelForCausalLM.from_pretrained(
            staging_dir,
            export=True,
            compile=False,
            load_in_8bit=False,
            quantization_config=quantization_config,
            use_cache=False, # Stateless
            trust_remote_code=False,
            attn_implementation="eager"
        )

        model.save_pretrained(intermediate_dir)

        # VERIFY SIZE
        bin_path = os.path.join(intermediate_dir, "openvino_model.bin")
        if os.path.exists(bin_path):
            size_gb = os.path.getsize(bin_path) / (1024**3)
            logger.info(f"    > Intermediate Model Size: {size_gb:.2f} GB")
            if size_gb < 1.0:
                raise RuntimeError(f"Exported model is too small ({size_gb:.2f} GB). Graph tracing failed.")

        # Save Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(staging_dir, trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(staging_dir, trust_remote_code=True)

        tokenizer.save_pretrained(output_dir)

        del model
        gc.collect()

        # 3. Stage 2: Reshape to Static [1, 4096]
        logger.info(f">>> [Bake] Stage 2: Reshaping to [1, {STATIC_SEQ_LEN}]...")
        core = ov.Core()
        xml_path = os.path.join(intermediate_dir, "openvino_model.xml")
        ov_model = core.read_model(xml_path)

        new_shapes = {}
        for input_node in ov_model.inputs:
            name = input_node.any_name
            shape = input_node.get_partial_shape()
            logger.info(f"    > Input: {name} {shape}")

            if len(shape) == 2: # input_ids, attention_mask
                new_shapes[name] = ov.PartialShape([STATIC_BATCH_SIZE, STATIC_SEQ_LEN])
                logger.info(f"      -> Locking to [{STATIC_BATCH_SIZE}, {STATIC_SEQ_LEN}]")

        if new_shapes:
            ov_model.reshape(new_shapes)
            ov_model.validate_nodes_and_infer_types()

        # 4. Save Final
        final_xml = os.path.join(output_dir, "openvino_model.xml")
        logger.info(f">>> [Bake] Saving final static model to {output_dir}...")
        ov.save_model(ov_model, final_xml)

        # Copy Configs
        shutil.copy2(os.path.join(intermediate_dir, "config.json"), os.path.join(output_dir, "config.json"))
        # Copy others if exist
        for f_name in os.listdir(intermediate_dir):
            if f_name.endswith(".json") and "config" in f_name and not os.path.exists(os.path.join(output_dir, f_name)):
                shutil.copy2(os.path.join(intermediate_dir, f_name), os.path.join(output_dir, f_name))

        # Cleanup
        shutil.rmtree(intermediate_dir)
        logger.info(">>> [Bake] DONE.")

    except Exception as e:
        logger.error(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging_dir", default="./model_staging_qwen")
    parser.add_argument("--output_dir", default="./models/qwen3_int4")
    parser.add_argument("--config", default="src/python/nncf_config.json")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    bake_qwen(args.staging_dir, args.output_dir, args.config)
