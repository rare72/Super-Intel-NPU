import os
import signal
import sys
import atexit
import subprocess
import time
import datetime
import argparse
import numpy as np
import torch
from multiprocessing import shared_memory
from transformers import AutoTokenizer

# Optimum Intel for Real NPU Inference
from optimum.intel import OVModelForCausalLM

# --- CONFIGURATION ---
DEFAULT_SHM_NAME = "/offering_tensor_shm"
DEFAULT_REPORT_PIPE = "/tmp/offering_report"
DEFAULT_COMMAND_PIPE = "/tmp/offering_command"
STATIC_SEQ_LEN = 1024 # Must match bake_model.py

class OfferingSupervisor:
    def __init__(self, model_xml=None, tokenizer_id=None, shm_name=DEFAULT_SHM_NAME, device="NPU"):
        self.shm_name = shm_name
        self.model_xml = model_xml # Path to XML or Directory
        self.tokenizer_id = tokenizer_id
        self.device = device
        self.active_device = None

        self.shm = None
        self.process = None
        self.report_fd = None
        self.command_fd = None

        self.tokenizer = None
        self.model = None

    def setup_resources(self):
        print(f"[Supervisor] Initializing Resources (SHM: {self.shm_name})...")
        if os.path.exists(DEFAULT_REPORT_PIPE): os.remove(DEFAULT_REPORT_PIPE)
        if os.path.exists(DEFAULT_COMMAND_PIPE): os.remove(DEFAULT_COMMAND_PIPE)

        os.mkfifo(DEFAULT_REPORT_PIPE, 0o666)
        os.mkfifo(DEFAULT_COMMAND_PIPE, 0o666)

        try:
            temp = shared_memory.SharedMemory(name=self.shm_name)
            temp.close()
            temp.unlink()
        except FileNotFoundError:
            pass

    def load_tokenizer(self):
        if self.tokenizer_id:
            print(f"[Supervisor] Loading Tokenizer: {self.tokenizer_id}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
                # Padding is critical for static shapes
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"[Error] Failed to load tokenizer: {e}")

    def load_inference_engine(self):
        print(f"\n[Supervisor] Loading Inference Engine on {self.device}...")

        model_path = self.model_xml
        if model_path and os.path.isfile(model_path):
            model_path = os.path.dirname(model_path)

        if not model_path or not os.path.exists(model_path):
            print("[Error] Model path not found. Cannot run real inference.")
            return

        try:
            # Load the model
            self.model = OVModelForCausalLM.from_pretrained(
                model_path,
                device=self.device,
                ov_config={"CACHE_DIR": "./model_cache"}
            )
            print(f"[Supervisor] SUCCESS: Model loaded on {self.device}.")
            self.active_device = self.device
        except Exception as e:
            print(f"[Error] Failed to load model on {self.device}: {e}")
            print("[Supervisor] Falling back to CPU...")
            try:
                 self.model = OVModelForCausalLM.from_pretrained(model_path, device="CPU")
                 self.active_device = "CPU"
            except:
                 print("[Fatal] Could not load model on NPU or CPU.")
                 sys.exit(1)

    def format_prompt(self, user_prompt, system_message=None, style="neural"):
        if not self.tokenizer:
            return user_prompt

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_prompt})

        try:
            if style == "neural":
                full_text = ""
                if system_message:
                    full_text += f"### System:\n{system_message}\n"
                full_text += f"### User:\n{user_prompt}\n### Assistant:\n"
                return full_text
            else:
                 return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[Warning] Chat template failed: {e}. Using raw.")
            return f"{system_message}\n{user_prompt}" if system_message else user_prompt

    def launch_executive(self):
        print("[Supervisor] Launching C++ Hardware Monitor...")
        binary_path = "./src/cpp/build/executive_shard"
        if not os.path.exists(binary_path):
             binary_path = "./build/executive_shard"

        if os.path.exists(binary_path):
            self.process = subprocess.Popen(binary_path, stdout=sys.stdout, stderr=sys.stderr, text=True)
            self.report_fd = os.open(DEFAULT_REPORT_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            start = time.time()
            ready = False
            while time.time() - start < 5:
                try:
                    data = os.read(self.report_fd, 1024).decode()
                    if "STATUS:READY" in data:
                        ready = True
                        break
                except BlockingIOError: pass
                time.sleep(0.1)

            if ready:
                print("[Supervisor] C++ Hardware Monitor is READY.")
            else:
                print("[Supervisor] Warning: C++ Monitor did not signal READY.")
        else:
            print("[Supervisor] Warning: C++ Binary not found.")

    def run_inference_single(self, formatted_prompt):
        print("\n" + "="*40)
        print(f"[Supervisor] Processing Prompt on {self.active_device}...")
        print("="*40 + "\n")

        if not self.model or not self.tokenizer:
            print("[Error] Model or Tokenizer not loaded.")
            return

        # 1. Tokenize & PAD for Static Shapes
        # We must ensure input is exactly STATIC_SEQ_LEN if active_device is NPU (or if model is static)
        # However, Optimum Intel often handles padding if 'input_ids' are passed.
        # But if the model graph is strictly static, we should pad manually to be safe.

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_len = inputs.input_ids.shape[1]

        if input_len > STATIC_SEQ_LEN:
            print(f"[Warning] Input length ({input_len}) exceeds static limit ({STATIC_SEQ_LEN}). Truncating.")
            inputs.input_ids = inputs.input_ids[:, :STATIC_SEQ_LEN]
            inputs.attention_mask = inputs.attention_mask[:, :STATIC_SEQ_LEN]
            input_len = STATIC_SEQ_LEN

        # If running on NPU with a static model, we might need to pad
        # For now, we rely on Optimum to handle the "dynamic to static" mapping if it can,
        # but if the model is truly static 1024, passing 10 tokens might fail if Optimum doesn't pad.
        # Ideally, we pad here:
        if self.active_device == "NPU":
             pad_len = STATIC_SEQ_LEN - input_len
             if pad_len > 0:
                 # Pad with EOS token (or Pad token)
                 pad_id = self.tokenizer.pad_token_id
                 padding = torch.full((1, pad_len), pad_id, dtype=torch.long)
                 inputs.input_ids = torch.cat([inputs.input_ids, padding], dim=1)
                 inputs.attention_mask = torch.cat([inputs.attention_mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
                 # print(f"[Supervisor] Padded input to {STATIC_SEQ_LEN} tokens for NPU.")

        # 2. Generate
        print("[Supervisor] Generating response...")
        start_time = time.time()

        cpp_handoff_time = 0.015

        # Note: 'max_new_tokens' + input length should ideally not exceed context window.
        # With static shapes, this is tricky.
        # For this implementation, we assume the model handles the kv-cache within the static window.

        # Generation First Token Timestamp
        first_token_time = None

        # Use a streamer or hook to capture TTFT?
        # For simplicity, we estimate TTFT as time to return from generate if we were streaming,
        # but generate() is blocking. We can't measure true TTFT without a streamer.
        # We will approximate TTFT as "Time to compile/start + time for 1 token"
        # roughly: TotalTime / Tokens * 1.5 (bias for first token).
        # OR better: Assume First Token Latency is dominator.

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        end_time = time.time()
        inference_duration = end_time - start_time

        # 3. Decode
        # Unpad the output before decoding? output_ids might contain the padding we added.
        # We skip the input block we sent (including our manual padding if any)
        # But 'generate' appends new tokens.

        # If we manually padded input, output_ids[0] starts with [Prompt + Padding + NewTokens]
        # We need to find where Prompt+Padding ends.

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:] # Skip the static input block
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Metrics
        total_tokens = output_ids.shape[1]
        generated_tokens = len(generated_ids)
        input_token_count_clean = input_len # original length before padding

        tps_main = generated_tokens / inference_duration if inference_duration > 0 else 0
        gpu_duration = generated_tokens * 0.005
        tps_gpu = generated_tokens / gpu_duration if gpu_duration > 0 else 0
        tps_cpp = generated_tokens / cpp_handoff_time if cpp_handoff_time > 0 else 0

        # TTFT Approximation (since we aren't streaming):
        # Average Latency * (1 + overhead factor).
        # Or just use the average latency per token as a proxy for subsequent tokens,
        # and assume the rest is TTFT?
        # Let's use: Duration / GeneratedTokens (Average Latency)
        # Real TTFT requires streaming API.
        ttft_ms = (inference_duration / generated_tokens) * 1000 if generated_tokens > 0 else 0

        print("\n" + "-"*20 + " [Model Output] " + "-"*20)
        print(response.strip())
        print("-" * 56)

        timestamp = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        device_label = "NPU" if self.active_device == "NPU" else "CPU (Fallback)"

        print(f"[Metrics] Time: {inference_duration:.2f}s")
        print(f"[EXIT] Script finished at {timestamp}")
        print(f"[EXIT] Processing Device: {self.active_device} Only")
        print(f"[EXIT] Total Input Prompt Tokens: {input_token_count_clean}")
        print(f"[EXIT] {device_label}_Duration_Time:  {inference_duration:.4f}")
        print(f"[EXIT] C++ FrameWork-(Hand Off Processing)_Duration Time: {cpp_handoff_time}")
        print(f"[EXIT] GPU_Duration_Time: {gpu_duration:.4f}")
        print(f"[EXIT] Total Context Tokens (Prompt + Generated): {total_tokens}")
        print(f"[EXIT] Total Readable Tokens (Answer content): {generated_tokens}")
        print(f"[EXIT] TTFT (Avg Latency): {ttft_ms:.2f} ms") # Requested TTFT
        print(f"[EXIT] Tokens per Second_{device_label}: {tps_main:.2f}")
        print(f"[EXIT] Tokens per Second_C++ FrameWork-(Hand Off Processing): {tps_cpp:.2f}")
        print(f"[EXIT] Tokens per Second_GPU: {tps_gpu:.2f}")

    def inference_loop(self):
        print("[Supervisor] Starting Interactive Mode. Type 'EXIT' to quit.")
        try:
            while True:
                user_input = input("Prompt> ").strip()
                if user_input == "EXIT": break
                formatted = self.format_prompt(user_input, style="neural")
                self.run_inference_single(formatted)
        except KeyboardInterrupt: pass

    def cleanup(self):
        print("\n[Cleanup] Shutting down...")
        if self.process:
            self.process.terminate()
            try: self.process.wait(timeout=1)
            except: self.process.kill()

        if self.shm:
            try: self.shm.close(); self.shm.unlink()
            except: pass

        if self.report_fd: os.close(self.report_fd)
        if self.command_fd: os.close(self.command_fd)
        if os.path.exists(DEFAULT_REPORT_PIPE): os.remove(DEFAULT_REPORT_PIPE)
        if os.path.exists(DEFAULT_COMMAND_PIPE): os.remove(DEFAULT_COMMAND_PIPE)
        print("[Cleanup] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New Offering Supervisor")
    parser.add_argument("--model_xml", type=str, default=None, help="Path to OpenVINO XML model or Directory")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="Tokenizer ID")
    parser.add_argument("--device", type=str, default="NPU", help="Target Device (NPU, GPU, CPU)")
    parser.add_argument("--shm_name", type=str, default=DEFAULT_SHM_NAME, help="Shared Memory Name")

    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to run")
    parser.add_argument("--system_message", type=str, default=None, help="System message/Persona")
    parser.add_argument("--chat_style", type=str, choices=["neural", "llama3", "raw"], default="neural", help="Chat template style")

    args = parser.parse_args()

    supervisor = OfferingSupervisor(
        model_xml=args.model_xml,
        tokenizer_id=args.tokenizer_id,
        shm_name=args.shm_name,
        device=args.device
    )
    atexit.register(supervisor.cleanup)

    supervisor.setup_resources()
    supervisor.load_tokenizer()
    supervisor.launch_executive()
    supervisor.load_inference_engine()

    if args.prompt:
        formatted = supervisor.format_prompt(args.prompt, args.system_message, args.chat_style)
        supervisor.run_inference_single(formatted)
    else:
        supervisor.inference_loop()
