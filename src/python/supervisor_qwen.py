import os
import sys
import argparse
import time
import datetime
import numpy as np
import torch
import atexit
import subprocess
import logging
from transformers import AutoTokenizer
import openvino as ov

# --- CONFIGURATION ---
STATIC_SEQ_LEN = 4096

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("QwenSupervisor")

class QwenSupervisor:
    def __init__(self, model_path, device="NPU", turbo=False):
        self.model_path = model_path
        self.device = device
        self.turbo = turbo
        self.tokenizer = None
        self.request = None
        self.input_map = {}

    def load(self):
        logger.info(f"Loading Tokenizer from {self.model_path}...")
        try:
            # Attempt to fix the specific Mistral-regex warning if supported
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            # Fallback for versions/classes that don't support the flag
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"

        # Diagnostics
        logger.info("--- SYSTEM DIAGNOSTICS ---")
        try:
            subprocess.run(['free', '-h'], check=False)
        except Exception:
            pass

        xml_path = os.path.join(self.model_path, "openvino_model.xml")
        bin_path = os.path.join(self.model_path, "openvino_model.bin")

        if os.path.exists(xml_path) and os.path.exists(bin_path):
            xml_mb = os.path.getsize(xml_path) / (1024*1024)
            bin_gb = os.path.getsize(bin_path) / (1024**3)
            logger.info(f"Model files found: XML={xml_mb:.2f}MB, BIN={bin_gb:.2f}GB")

            # CRITICAL INTEGRITY CHECK
            if bin_gb < 2.0:
                logger.error(f"CRITICAL: Model binary is too small ({bin_gb:.2f} GB). Expected > 4.0 GB.")
                logger.error("The model graph was likely truncated during export.")
                logger.error("SOLUTION: Please delete the model directory and re-run 'bake_qwen.py'.")
                sys.exit(1)
        else:
            logger.error(f"CRITICAL: Model files not found at {self.model_path}")
            sys.exit(1)

        logger.info(f"Loading Model to {self.device} (This may take 5-15 mins for NPU Compilation)...")
        sys.stdout.flush()

        core = ov.Core()

        # Enable Verbose OpenVINO Logging
        logger.info("Enabling OpenVINO Verbose Logging...")
        try:
            core.set_property({"LOG_LEVEL": "LOG_INFO"})
            if self.device == "NPU":
                core.set_property("NPU", {"LOG_LEVEL": "LOG_DEBUG"})
                turbo_val = "YES" if self.turbo else "NO"
                logger.info(f"Setting NPU_TURBO={turbo_val}")
                core.set_property("NPU", {"NPU_TURBO": turbo_val})
        except Exception as e:
            logger.warning(f"Could not set log properties: {e}")

        core.set_property({"CACHE_DIR": "./model_cache_qwen"})

        model = core.compile_model(xml_path, self.device)
        self.request = model.create_infer_request()

        # Map inputs
        for i in model.inputs:
            self.input_map[i.any_name] = i.element_type
            print(f"  > Input: {i.any_name} ({i.element_type})")

        # Warmup
        print("[Qwen] Warming up NPU (Stabilizing Driver)...")
        dummy_input = {
            "input_ids": np.full((1, STATIC_SEQ_LEN), self.tokenizer.pad_token_id, dtype=np.int64),
            "attention_mask": np.zeros((1, STATIC_SEQ_LEN), dtype=np.int64)
        }
        if "position_ids" in self.input_map:
             dummy_input["position_ids"] = np.arange(0, STATIC_SEQ_LEN, dtype=np.int64).reshape(1, -1)

        self.request.infer(dummy_input)
        print("[Qwen] Warmup Complete.")

    def infer(self, prompt):
        print("-" * 40)
        print(f"PROMPT: {prompt}")
        print("-" * 40)

        # 1. Format with Thinking Template
        # Qwen's template usually handles system prompts.
        # We explicitly inject 'thinking' enabling if needed, but user just wants to SEE it.
        # The default template usually hides it? We'll use apply_chat_template.
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You must always think before you speak. Output your thought process in <think> tags."},
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2. Tokenize & Pad
        enc = self.tokenizer(text_input, return_tensors="pt")
        input_ids = enc.input_ids
        attn_mask = enc.attention_mask

        curr_len = input_ids.shape[1]
        print(f"[Qwen] Input Length: {curr_len} / {STATIC_SEQ_LEN}")

        if curr_len > STATIC_SEQ_LEN:
            print("!! Error: Prompt too long for static window.")
            return

        # 3. Generation Loop
        start_time = time.time()
        generated = []

        print("[Qwen] Streaming output...")

        for _ in range(STATIC_SEQ_LEN - curr_len):
            # Pad to [1, 4096]
            pad_len = STATIC_SEQ_LEN - input_ids.shape[1]

            # Create padded tensors
            pad_ids = torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_msk = torch.zeros((1, pad_len), dtype=torch.long)

            full_ids = torch.cat([input_ids, pad_ids], dim=1)
            full_msk = torch.cat([attn_mask, pad_msk], dim=1)

            # Run Inference
            inputs = {
                "input_ids": full_ids.numpy(),
                "attention_mask": full_msk.numpy()
            }
            # Handle position_ids if required by model
            if "position_ids" in self.input_map:
                pos_ids = torch.arange(0, STATIC_SEQ_LEN, dtype=torch.long).unsqueeze(0)
                inputs["position_ids"] = pos_ids.numpy()

            self.request.infer(inputs)

            # Sample
            logits = torch.from_numpy(self.request.get_output_tensor(0).data)
            next_token_logits = logits[:, input_ids.shape[1]-1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Decode & Print
            word = self.tokenizer.decode(next_token.item())
            print(word, end="", flush=True)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones((1,1), dtype=torch.long)], dim=1)

            if next_token.item() in [self.tokenizer.eos_token_id, 151645]: # 151645 is often <|im_end|>
                break

        print("\n[Qwen] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models/qwen3_int4")
    parser.add_argument("--prompt", default="Why is the sky blue?")
    parser.add_argument("--npu_turbo", action="store_true", help="Enable NPU Turbo Mode (Experimental)")
    args = parser.parse_args()

    sup = QwenSupervisor(args.model_dir, turbo=args.npu_turbo)
    sup.load()
    sup.infer(args.prompt)
