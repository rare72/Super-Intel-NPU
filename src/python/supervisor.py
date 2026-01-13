import os
import signal
import sys
import atexit
import subprocess
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import shared_memory
from transformers import AutoTokenizer

# --- CONFIGURATION ---
DEFAULT_SHM_NAME = "/offering_tensor_shm"
DEFAULT_REPORT_PIPE = "/tmp/offering_report"
DEFAULT_COMMAND_PIPE = "/tmp/offering_command"
TENSOR_SIZE_BYTES = 4096 * 4  # 4096 float32s

class OfferingSupervisor:
    def __init__(self, model_xml=None, tokenizer_id=None, shm_name=DEFAULT_SHM_NAME):
        self.shm_name = shm_name
        self.model_xml = model_xml
        self.tokenizer_id = tokenizer_id
        self.shm = None
        self.process = None
        self.report_fd = None
        self.command_fd = None
        self.gpu_shard = None
        self.tokenizer = None

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
                # Ensure pad token is set for batched inference if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"[Error] Failed to load tokenizer: {e}")

    def format_prompt(self, user_prompt, system_message=None, style="neural"):
        """
        Applies the specific chat template logic.
        """
        if not self.tokenizer:
            return user_prompt # Fallback

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_prompt})

        # Neural Chat specific manual formatting (if tokenizer template is tricky or we want explicit control)
        # However, we verified the tokenizer has a chat_template. Let's try to use apply_chat_template first.
        # But user explicitly asked about the format.

        try:
            # Check if style overrides are needed
            if style == "neural":
                # Explicit fallback if apply_chat_template fails or we want to force the ### format
                full_text = ""
                if system_message:
                    full_text += f"### System:\n{system_message}\n"
                full_text += f"### User:\n{user_prompt}\n### Assistant:\n"
                return full_text
            elif style == "llama3":
                 # Llama 3 usually handles system messages via apply_chat_template correctly
                 return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                 # Raw or auto
                 return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[Warning] Chat template application failed: {e}. Falling back to raw concatenation.")
            return f"{system_message}\n{user_prompt}" if system_message else user_prompt

    def launch_executive(self):
        print("[Supervisor] Launching C++ Executive...")
        binary_path = "./src/cpp/build/executive_shard"
        if not os.path.exists(binary_path):
             binary_path = "./build/executive_shard"

        if not os.path.exists(binary_path):
            print(f"[Error] C++ Binary not found at {binary_path}. Please compile first.")
            sys.exit(1)

        cmd = [binary_path]
        self.process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

        print("[Supervisor] Waiting for Executive READY signal...")
        self.report_fd = os.open(DEFAULT_REPORT_PIPE, os.O_RDONLY | os.O_NONBLOCK)

        ready = False
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                data = os.read(self.report_fd, 1024).decode()
                if "STATUS:READY" in data:
                    ready = True
                    break
            except BlockingIOError:
                pass
            time.sleep(0.1)

        if not ready:
             print("[Error] Timed out waiting for C++ Executive.")
             self.cleanup()
             sys.exit(1)

        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.tensor_view = np.ndarray((4096,), dtype=np.float32, buffer=self.shm.buf)
            print("[Supervisor] Shared Memory Attached.")
        except FileNotFoundError:
             print(f"[Error] Shared memory {self.shm_name} not found.")
             self.cleanup()
             sys.exit(1)

        self.command_fd = os.open(DEFAULT_COMMAND_PIPE, os.O_WRONLY)

    def load_gpu_shard(self):
        print("[Supervisor] Loading GPU Shard (PyTorch)...")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[Supervisor] GPU Shard loaded on CUDA.")
        else:
            print("[Supervisor] Warning: CUDA not available. Running in CPU mode for simulation.")
            self.device = torch.device("cpu")

    def run_inference_single(self, formatted_prompt):
        print("\n" + "="*40)
        print(f"[Supervisor] Processing Prompt:\n{formatted_prompt}")
        print("="*40 + "\n")

        # 1. Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer(formatted_prompt, return_tensors="np").input_ids
            print(f"[Supervisor] Tokenized Input Shape: {input_ids.shape}")
            # In a real implementation, we would write these input_ids to a shared buffer for the NPU
            # or send them via a pipe. For this skeleton, we assume the NPU knows what to do
            # (or we just trigger the signal).

        # 2. Trigger NPU
        print("[Supervisor] Sending PROCESS command to NPU...")
        os.write(self.command_fd, b"PROCESS")

        # 3. Wait for NPU Result (Hidden States)
        print("[Supervisor] Waiting for NPU result (Hidden States)...")
        while True:
            try:
                data = os.read(self.report_fd, 1024).decode()
                if "STATUS:DATA_READY" in data:
                    break
            except BlockingIOError:
                pass
            time.sleep(0.01)

        # 4. Read Shared Memory (Zero Copy)
        val = self.tensor_view[0]
        print(f"[Supervisor] Received Tensor Data from NPU! First Value: {val}")

        # 5. GPU Handoff
        print("[Supervisor] Handoff to GPU Shard complete.")
        print("[Result] (Simulated output based on dummy NPU data)")

    def inference_loop(self):
        print("[Supervisor] Starting Interactive Mode. Type 'EXIT' to quit.")
        try:
            while True:
                user_input = input("Prompt> ").strip()
                if user_input == "EXIT":
                    os.write(self.command_fd, b"EXIT")
                    break

                # Use default formatting for interactive mode
                formatted = self.format_prompt(user_input, style="neural")
                self.run_inference_single(formatted)

        except KeyboardInterrupt:
            pass

    def cleanup(self):
        print("\n[Cleanup] Shutting down...")
        if self.process:
            self.process.terminate()
            try: self.process.wait(timeout=1)
            except: self.process.kill()

        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except: pass

        if self.report_fd: os.close(self.report_fd)
        if self.command_fd: os.close(self.command_fd)

        if os.path.exists(DEFAULT_REPORT_PIPE): os.remove(DEFAULT_REPORT_PIPE)
        if os.path.exists(DEFAULT_COMMAND_PIPE): os.remove(DEFAULT_COMMAND_PIPE)
        print("[Cleanup] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New Offering Supervisor")
    parser.add_argument("--model_xml", type=str, default=None, help="Path to OpenVINO XML model")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="Tokenizer ID")
    parser.add_argument("--shm_name", type=str, default=DEFAULT_SHM_NAME, help="Shared Memory Name")

    # Prompting Arguments
    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to run")
    parser.add_argument("--system_message", type=str, default=None, help="System message/Persona")
    parser.add_argument("--chat_style", type=str, choices=["neural", "llama3", "raw"], default="neural", help="Chat template style")

    args = parser.parse_args()

    supervisor = OfferingSupervisor(
        model_xml=args.model_xml,
        tokenizer_id=args.tokenizer_id,
        shm_name=args.shm_name
    )
    atexit.register(supervisor.cleanup)

    supervisor.setup_resources()
    supervisor.load_tokenizer() # Load this early
    supervisor.launch_executive()
    supervisor.load_gpu_shard()

    if args.prompt:
        # Single shot mode
        formatted = supervisor.format_prompt(args.prompt, args.system_message, args.chat_style)
        supervisor.run_inference_single(formatted)
        # Cleanup happens via atexit
    else:
        # Interactive mode
        supervisor.inference_loop()
