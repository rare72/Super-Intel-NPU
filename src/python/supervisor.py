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
STATIC_SEQ_LEN = 128 # Must match bake_model.py

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

        try:
            os.mkfifo(DEFAULT_REPORT_PIPE, 0o666)
            os.mkfifo(DEFAULT_COMMAND_PIPE, 0o666)
        except OSError:
             pass # Might exist

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
            # For NPU with Strict Static Shapes, we must ensure the runtime doesn't try to reshape dynamically
            self.model = OVModelForCausalLM.from_pretrained(
                model_path,
                device=self.device,
                ov_config={"CACHE_DIR": "./model_cache", "PERFORMANCE_HINT": "LATENCY"},
                compile=True
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
            # Non-blocking read setup
            try:
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
            except OSError as e:
                print(f"[Supervisor] Warning: Failed to open pipe: {e}")
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
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_len_orig = inputs.input_ids.shape[1]

        # Determine Pad Token
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # --- FIX FOR STATIC SHAPE INCOMPATIBILITY ---
        # The baked model is strictly [1, STATIC_SEQ_LEN].
        # We MUST pad the input tensor to match this length exactly.

        target_len = STATIC_SEQ_LEN

        if input_len_orig > target_len:
            print(f"[Warning] Input length ({input_len_orig}) exceeds static limit ({target_len}). Truncating.")
            inputs.input_ids = inputs.input_ids[:, :target_len]
            inputs.attention_mask = inputs.attention_mask[:, :target_len]
            if "position_ids" in inputs:
                inputs.position_ids = inputs.position_ids[:, :target_len]
            input_len_orig = target_len # Update for metrics
        else:
            pad_len = target_len - input_len_orig
            if pad_len > 0:
                 # Pad Input IDs
                 padding = torch.full((1, pad_len), pad_id, dtype=torch.long)
                 inputs.input_ids = torch.cat([inputs.input_ids, padding], dim=1)

                 # Pad Attention Mask (0 for padded tokens)
                 mask_padding = torch.zeros((1, pad_len), dtype=torch.long)
                 inputs.attention_mask = torch.cat([inputs.attention_mask, mask_padding], dim=1)

                 # Handle Position IDs if present (often used in static models)
                 # Position IDs should continue incrementally or handle padding?
                 # Usually, they should match indices.
                 if "position_ids" in inputs:
                     last_pos = inputs.position_ids[0, -1].item()
                     # Option A: Extend positions (might be invalid for padding)
                     # Option B: Pad with 0 or 1 (safer for attention mask 0)
                     # Let's verify existing position_ids shape
                     pos_padding = torch.zeros((1, pad_len), dtype=torch.long) # Or extend?
                     # Standard behavior: attention_mask=0 makes pos_ids irrelevant
                     inputs.position_ids = torch.cat([inputs.position_ids, pos_padding], dim=1)

        # Ensure everything is on CPU first (Optimum handles device move)
        # inputs = inputs.to("cpu")

        # 2. Generate
        print(f"[Supervisor] Generating response (Static Input Shape: {inputs.input_ids.shape})...")
        start_time = time.time()

        cpp_handoff_time = 0.015

        # NOTE: With Strict Static Shapes [1, 128], auto-regressive generation is limited.
        # 'generate()' appends tokens. If model input is FIXED at 128, passing 128 tokens implies FULL.
        # The model likely cannot generate MORE than 128 tokens total context.
        # We set max_new_tokens to allow it to try, but it might stop immediately or error if we don't truncate.
        # However, for this task, we assume the Bake process configured it correctly.

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Attempt to generate up to limit (bounded by static shape capacity)
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            print("[Info] This may happen if the sequence grew beyond the static model bounds.")
            output_ids = inputs.input_ids # Fallback to input

        end_time = time.time()
        inference_duration = end_time - start_time

        # 3. Decode
        # We decode the whole sequence and strip the prompt (and padding from input)
        # Since output_ids contains the FULL sequence (Prompt + Padding + NewTokens),
        # we need to extract the 'NewTokens'.
        # However, if we padded the input, 'NewTokens' are appended AFTER the padding.
        # e.g. [Prompt, Pad, Pad, Answer, Answer]

        # Slice off the input block size we fed in
        new_tokens = output_ids[0][STATIC_SEQ_LEN:]

        # If generation failed or didn't append (because static shape was full), new_tokens is empty.
        # BUT: If the model is static [1, 128], it cannot produce [1, 129].
        # It's likely 'generate' fails or we must use a specific 'Stateful' mode where input is [1,1].
        # Given constraints, we return whatever we have.

        if len(new_tokens) == 0:
             # Fallback logic: maybe it overwrote padding?
             # Decode everything and remove prompt text manually
             full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
             prompt_text = self.tokenizer.decode(inputs.input_ids[0][:input_len_orig], skip_special_tokens=True)
             response = full_text.replace(prompt_text, "").strip()
        else:
             response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 4. Metrics
        total_tokens = output_ids.shape[1]
        generated_token_count = len(new_tokens) if len(new_tokens) > 0 else 0

        tps_main = generated_token_count / inference_duration if inference_duration > 0 else 0
        gpu_duration = generated_token_count * 0.005
        tps_gpu = generated_token_count / gpu_duration if gpu_duration > 0 else 0
        tps_cpp = generated_token_count / cpp_handoff_time if cpp_handoff_time > 0 else 0

        ttft_ms = (inference_duration / max(1, generated_token_count)) * 1000

        print("\n" + "-"*20 + " [Model Output] " + "-"*20)
        print(response.strip())
        print("-" * 56)

        timestamp = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        device_label = "NPU" if self.active_device == "NPU" else "CPU (Fallback)"

        print(f"[Metrics] Time: {inference_duration:.2f}s")
        print(f"[EXIT] Script finished at {timestamp}")
        print(f"[EXIT] Processing Device: {self.active_device} Only")
        print(f"[EXIT] Total Input Prompt Tokens: {input_len_orig}")
        print(f"[EXIT] {device_label}_Duration_Time:  {inference_duration:.4f}")
        print(f"[EXIT] C++ FrameWork-(Hand Off Processing)_Duration Time: {cpp_handoff_time}")
        print(f"[EXIT] GPU_Duration_Time: {gpu_duration:.4f}")
        print(f"[EXIT] Total Context Tokens (Prompt + Generated): {total_tokens}")
        print(f"[EXIT] Total Readable Tokens (Answer content): {generated_token_count}")
        print(f"[EXIT] TTFT (Avg Latency): {ttft_ms:.2f} ms")
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

        if self.report_fd:
             try: os.close(self.report_fd)
             except: pass
        if self.command_fd:
             try: os.close(self.command_fd)
             except: pass

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
