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

# --- CONFIGURATION ---
# Default Configuration
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

    def setup_resources(self):
        print(f"[Supervisor] Initializing Resources (SHM: {self.shm_name})...")

        # Cleanup any stale pipes
        if os.path.exists(DEFAULT_REPORT_PIPE): os.remove(DEFAULT_REPORT_PIPE)
        if os.path.exists(DEFAULT_COMMAND_PIPE): os.remove(DEFAULT_COMMAND_PIPE)

        os.mkfifo(DEFAULT_REPORT_PIPE, 0o666)
        os.mkfifo(DEFAULT_COMMAND_PIPE, 0o666)

        # We rely on C++ to create the Shared Memory, but we can try to cleanup stale one first
        try:
            temp = shared_memory.SharedMemory(name=self.shm_name)
            temp.close()
            temp.unlink()
            print("[Supervisor] Cleaned up stale shared memory.")
        except FileNotFoundError:
            pass

    def launch_executive(self):
        print("[Supervisor] Launching C++ Executive...")
        # Assuming binary is in the current directory or build directory
        # Adjust path as necessary
        binary_path = "./src/cpp/build/executive_shard"
        if not os.path.exists(binary_path):
             binary_path = "./build/executive_shard" # Fallback

        if not os.path.exists(binary_path):
            print(f"[Error] C++ Binary not found at {binary_path}. Please compile first.")
            sys.exit(1)

        # In a real implementation, we might pass the model path to the C++ binary via args
        # e.g., [binary_path, "--model", self.model_xml]
        # For now, we assume the C++ binary is generic or uses a config.
        cmd = [binary_path]
        if self.model_xml:
            # We don't implement arg parsing in C++ yet, but this shows intent
            # cmd.extend(["--model", self.model_xml])
            print(f"[Supervisor] Note: Targeting model {self.model_xml} for NPU execution.")

        self.process = subprocess.Popen(cmd,
                                        stdout=sys.stdout,
                                        stderr=sys.stderr,
                                        text=True)

        # Wait for C++ to create SHM
        print("[Supervisor] Waiting for Executive READY signal...")

        # Open pipe for reading (blocking open until C++ opens for writing)
        self.report_fd = os.open(DEFAULT_REPORT_PIPE, os.O_RDONLY | os.O_NONBLOCK)

        # Poll for READY
        ready = False
        start_time = time.time()
        while time.time() - start_time < 10: # 10s timeout
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

        print(f"[Supervisor] Executive is READY. Attaching to Shared Memory {self.shm_name}...")
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            # Create a NumPy view of the raw RAM (Zero-Copy)
            self.tensor_view = np.ndarray((4096,), dtype=np.float32, buffer=self.shm.buf)
            print("[Supervisor] Shared Memory Attached.")
        except FileNotFoundError:
             print(f"[Error] Shared memory {self.shm_name} not found. C++ failed to create it?")
             self.cleanup()
             sys.exit(1)

        # Open Command Pipe
        self.command_fd = os.open(DEFAULT_COMMAND_PIPE, os.O_WRONLY)

    def load_gpu_shard(self):
        print("[Supervisor] Loading GPU Shard (PyTorch)...")
        if self.tokenizer_id:
             print(f"[Supervisor] Using Tokenizer: {self.tokenizer_id}")

        # Placeholder for actual GPU model loading
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # self.gpu_shard = ... load torch model ...
            print("[Supervisor] GPU Shard loaded on CUDA.")
        else:
            print("[Supervisor] Warning: CUDA not available. Running in CPU mode for simulation.")
            self.device = torch.device("cpu")

    def inference_loop(self):
        print("[Supervisor] Starting Inference Loop. Type 'EXIT' to quit, 'RUN' to simulate query.")
        try:
            while True:
                user_input = input("Command> ").strip()
                if user_input == "EXIT":
                    os.write(self.command_fd, b"EXIT")
                    break
                elif user_input == "RUN":
                    print("[Supervisor] Sending PROCESS command to NPU...")
                    os.write(self.command_fd, b"PROCESS")

                    # Wait for DATA_READY
                    print("[Supervisor] Waiting for NPU result...")
                    while True:
                        try:
                            data = os.read(self.report_fd, 1024).decode()
                            if "STATUS:DATA_READY" in data:
                                break
                        except BlockingIOError:
                            pass
                        time.sleep(0.01)

                    # Read Data (Zero Copy)
                    val = self.tensor_view[0]
                    print(f"[Supervisor] Received Tensor Data! Value[0]: {val}")

                    # Pass to GPU
                    # gpu_tensor = torch.from_numpy(self.tensor_view).to(self.device)
                    # result = self.gpu_shard(gpu_tensor)
                    print("[Supervisor] Handoff to GPU complete (Simulated).")

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
    parser.add_argument("--model_xml", type=str, default=None, help="Path to OpenVINO XML model (NPU Shard)")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="HuggingFace ID or path for Tokenizer")
    parser.add_argument("--shm_name", type=str, default=DEFAULT_SHM_NAME, help="Shared Memory Segment Name")

    args = parser.parse_args()

    supervisor = OfferingSupervisor(
        model_xml=args.model_xml,
        tokenizer_id=args.tokenizer_id,
        shm_name=args.shm_name
    )
    atexit.register(supervisor.cleanup)

    supervisor.setup_resources()
    supervisor.launch_executive()
    supervisor.load_gpu_shard()
    supervisor.inference_loop()
