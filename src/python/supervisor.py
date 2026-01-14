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
        self.use_cache_state = False # Track if we are in stateful or stateless mode

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

        # Strategy: Try Stateless (Static) first, as it's preferred for NPU stability.
        # If model was exported with --task text-generation-with-past, this will fail.
        # We catch that failure and switch to Stateful (Dynamic).

        configs_to_try = [False, True]
        success = False

        for try_cache in configs_to_try:
            try:
                print(f"[Supervisor] Attempting load with use_cache={try_cache}...")
                self.model = OVModelForCausalLM.from_pretrained(
                    model_path,
                    device=self.device,
                    ov_config={"CACHE_DIR": "./model_cache", "PERFORMANCE_HINT": "LATENCY"},
                    compile=True,
                    use_cache=try_cache
                )
                self.use_cache_state = try_cache
                success = True
                break # Loaded successfully
            except ValueError as e:
                # Catch specific Optimum error about cache mismatch
                if "use_cache" in str(e):
                    print(f"[Supervisor] Config mismatch detected: {e}")
                    print("[Supervisor] Switching cache mode and retrying...")
                    continue
                else:
                    print(f"[Error] Unexpected ValueError during load: {e}")
                    break
            except Exception as e:
                print(f"[Error] Failed to load model on {self.device}: {e}")
                break

        if success:
            # Inspect inputs to determine if beam_idx is required (only relevant for manual loop)
            try:
                self.model_input_names = [input.any_name for input in self.model.request.model_inputs]
                print(f"[Supervisor] Model Inputs: {self.model_input_names}")
            except:
                self.model_input_names = []

            print(f"[Supervisor] SUCCESS: Model loaded on {self.device}. Mode: {'Stateful' if self.use_cache_state else 'Stateless'}")
            self.active_device = self.device
        else:
            print("[Supervisor] Falling back to CPU...")
            try:
                 self.model = OVModelForCausalLM.from_pretrained(model_path, device="CPU", use_cache=True)
                 self.active_device = "CPU"
                 self.use_cache_state = True
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

    def run_inference(self, formatted_prompt):
        """
        Dispatches inference to the correct loop based on model configuration.
        """
        if self.use_cache_state:
            # Stateful Model (CLI Method) -> Use Standard Optimum Generate
            self.run_dynamic_stateful_inference(formatted_prompt)
        else:
            # Stateless Model (Bake Script) -> Use Custom Static Loop
            self.run_custom_static_inference(formatted_prompt)

    def run_dynamic_stateful_inference(self, formatted_prompt):
        """
        Uses optimum-intel's built-in generate() which handles KV cache management.
        Best for models exported via optimum-cli with --task text-generation-with-past.
        """
        print("\n" + "="*40)
        print(f"[Supervisor] Processing Prompt on {self.active_device} (Dynamic Stateful)...")
        print("="*40 + "\n")

        if not self.model or not self.tokenizer:
            return

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.active_device if self.active_device == "CPU" else "cpu")

        start_time = time.time()

        # Standard Optimum Generation
        # This handles past_key_values automatically
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

        end_time = time.time()
        inference_duration = end_time - start_time

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove input prompt from response if present
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):]

        self.print_metrics(response, inference_duration, len(output_ids[0]) - len(input_ids[0]), len(input_ids[0]))

    def run_custom_static_inference(self, formatted_prompt):
        """
        Manually executes inference loop ensuring strictly static inputs [1, 128]
        are passed to the NPU/CPU at every step, bypassing dynamic logic in Optimum.
        """
        print("\n" + "="*40)
        print(f"[Supervisor] Processing Prompt on {self.active_device} (Static Loop)...")
        print("="*40 + "\n")

        if not self.model or not self.tokenizer:
            return

        # 1. Tokenize Initial Input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Initial stats
        input_len = input_ids.shape[1]

        if input_len >= STATIC_SEQ_LEN:
            print(f"[Warning] Input ({input_len}) >= limit ({STATIC_SEQ_LEN}). Truncating.")
            input_ids = input_ids[:, :STATIC_SEQ_LEN]
            attention_mask = attention_mask[:, :STATIC_SEQ_LEN]
            input_len = STATIC_SEQ_LEN

        # 2. Loop Configuration
        if self.tokenizer.pad_token_id is not None:
             pad_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
             pad_id = self.tokenizer.eos_token_id
        else:
             pad_id = 0

        current_ids = input_ids
        current_mask = attention_mask

        generated_tokens = []
        max_new_tokens = STATIC_SEQ_LEN - input_len # Cannot grow beyond 128 total

        print(f"[Supervisor] Generating up to {max_new_tokens} tokens...")
        start_time = time.time()

        # 3. Static Generation Loop
        for i in range(max_new_tokens):
            current_len = current_ids.shape[1]
            if current_len >= STATIC_SEQ_LEN:
                break

            # A. PAD TO STRICT STATIC SHAPE [1, 128]
            pad_len = STATIC_SEQ_LEN - current_len

            # Pad IDs
            padding_ids = torch.full((1, pad_len), pad_id, dtype=torch.long)
            static_input_ids = torch.cat([current_ids, padding_ids], dim=1)

            # Pad Mask
            padding_mask = torch.zeros((1, pad_len), dtype=torch.long)
            static_attention_mask = torch.cat([current_mask, padding_mask], dim=1)

            # Create Position IDs (0..127)
            static_position_ids = torch.arange(0, STATIC_SEQ_LEN, dtype=torch.long).unsqueeze(0)

            # B. INFER (Direct OpenVINO Request)
            request = self.model.request

            inputs_dict = {
                "input_ids": static_input_ids.numpy().astype(np.int64),
                "attention_mask": static_attention_mask.numpy().astype(np.int64),
                "position_ids": static_position_ids.numpy().astype(np.int64)
            }

            # Conditionally add beam_idx
            if any("beam_idx" in name for name in self.model_input_names):
                inputs_dict["beam_idx"] = np.array([0], dtype=np.int32)

            # C. Run Inference
            try:
                request.infer(inputs_dict)
            except Exception as e:
                print(f"[Error] Inference failed at step {i}: {e}")
                break

            # D. Get Logits
            logits = torch.from_numpy(request.get_output_tensor(0).data)

            # Extract logit for the LAST REAL TOKEN
            next_token_logits = logits[:, current_len - 1, :]

            # E. Greedy Sample (Argmax)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # F. Check EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # G. Append
            generated_tokens.append(next_token_id.item())
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
            current_mask = torch.cat([current_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)

        end_time = time.time()
        inference_duration = end_time - start_time

        # 4. Final Decode
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.print_metrics(response, inference_duration, len(generated_tokens), input_len)

    def print_metrics(self, response, duration, generated_count, prompt_count):
        tps = generated_count / duration if duration > 0 else 0
        ttft = (duration / generated_count * 1000) if generated_count > 0 else 0

        print("\n" + "-"*20 + " [Model Output] " + "-"*20)
        print(response.strip())
        print("-" * 56)

        timestamp = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

        print(f"[Metrics] Time: {duration:.2f}s")
        print(f"[EXIT] Script finished at {timestamp}")
        print(f"[EXIT] Processing Device: {self.active_device} ({'Stateful' if self.use_cache_state else 'Stateless'})")
        print(f"[EXIT] Total Input Prompt Tokens: {prompt_count}")
        print(f"[EXIT] Total Generated Tokens: {generated_count}")
        print(f"[EXIT] TTFT (Avg Latency): {ttft:.2f} ms")
        print(f"[EXIT] Tokens per Second: {tps:.2f}")

    def inference_loop(self):
        print("[Supervisor] Starting Interactive Mode. Type 'EXIT' to quit.")
        try:
            while True:
                user_input = input("Prompt> ").strip()
                if user_input == "EXIT": break
                formatted = self.format_prompt(user_input, style="neural")
                self.run_inference(formatted)
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
        supervisor.run_inference(formatted)
    else:
        supervisor.inference_loop()
