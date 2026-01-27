import os
import sys
import argparse
import time
import logging
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import openvino as ov

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SupervisorQwenInt8")

class QwenInt8Supervisor:
    def __init__(self, model_path, device="NPU"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info(f"Loading Tokenizer from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"

        # Fallback Order Logic
        # If user requests NPU, try NPU -> GPU -> CPU
        devices_to_try = [self.device]
        if self.device == "NPU":
            if "GPU" not in devices_to_try: devices_to_try.append("GPU")
            if "CPU" not in devices_to_try: devices_to_try.append("CPU")

        logger.info(f"Loading Dynamic INT8 Model. Attempt Order: {devices_to_try}")

        for dev in devices_to_try:
            logger.info(f"Attempting load on: {dev}")

            # Configure OV Config for dynamic shapes
            ov_config = {"CACHE_DIR": "./model_cache_qwen_int8"}

            # For NPU specifically, we might need to be careful with dynamic shapes
            if dev == "NPU":
                logger.warning("WARNING: Running Dynamic Shapes on NPU. This may cause driver hangs/crashes.")
                ov_config["NPU_TURBO"] = "NO"

            try:
                self.model = OVModelForCausalLM.from_pretrained(
                    self.model_path,
                    device=dev,
                    ov_config=ov_config,
                    trust_remote_code=True
                )
                self.device = dev # Update to actual successful device
                logger.info(f"Model loaded successfully on {dev}.")
                return # Success
            except Exception as e:
                logger.error(f"Failed to load model on {dev}: {e}")
                if dev == devices_to_try[-1]:
                    logger.critical("All device attempts failed. Exiting.")
                    sys.exit(1)
                else:
                    logger.info("Retrying with next device in fallback chain...")

    def generate(self, prompt, max_new_tokens=100):
        logger.info(f"Prompt: {prompt}")

        # Apply Chat Template (Qwen style)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text_input, return_tensors="pt")

        logger.info("Generating...")
        start_time = time.time()

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        end_time = time.time()
        duration = end_time - start_time

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract just the new text if possible, or print full
        # The decode output usually includes prompt for causal LM unless handled
        logger.info("-" * 40)
        print(response)
        logger.info("-" * 40)

        num_tokens = len(output_ids[0])
        tps = num_tokens / duration
        logger.info(f"Metrics: {num_tokens} tokens in {duration:.2f}s ({tps:.2f} t/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models/pv1-qwen3_int8")
    parser.add_argument("--device", default="NPU", help="Device to run on (GPU, CPU, NPU)")
    parser.add_argument("--prompt", default="Explain quantum entanglement briefly.")
    args = parser.parse_args()

    sup = QwenInt8Supervisor(args.model_dir, device=args.device)
    sup.load()
    sup.generate(args.prompt)
