import os
import signal
import sys
import atexit
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import shared_memory

# --- CONFIGURATION ---
# Note: Python shared memory name does not need the leading slash on some systems,
# but shm_open in C++ usually requires it. Python's shared_memory module handles the platform differences.
# However, to be safe with cross-language, we align names carefully.
# In Python multiprocessing.shared_memory, passing name="name" usually maps to "/dev/shm/name" or "/name".
# C++ used "/offering_tensor_shm". Python should try "offering_tensor_shm" or "/offering_tensor_shm".
SHM_NAME = "/offering_tensor_shm"
REPORT_PIPE_PATH = "/tmp/offering_report"
COMMAND_PIPE_PATH = "/tmp/offering_command"
TENSOR_SIZE_BYTES = 4096 * 4  # 4096 float32s

class OfferingSupervisor:
    def __init__(self):
        self.shm = None
        self.process = None
        self.report_fd = None
        self.command_fd = None
        self.gpu_shard = None

    def setup_resources(self):
        print(f"[Supervisor] Initializing Resources...")

        # Cleanup any stale pipes
        if os.path.exists(REPORT_PIPE_PATH): os.remove(REPORT_PIPE_PATH)
        if os.path.exists(COMMAND_PIPE_PATH): os.remove(COMMAND_PIPE_PATH)

        os.mkfifo(REPORT_PIPE_PATH, 0o666)
        os.mkfifo(COMMAND_PIPE_PATH, 0o666)

        # We rely on C++ to create the Shared Memory, but we can try to cleanup stale one first
        try:
            temp = shared_memory.SharedMemory(name=SHM_NAME)
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

        self.process = subprocess.Popen([binary_path],
                                        stdout=sys.stdout,
                                        stderr=sys.stderr,
                                        text=True)

        # Wait for C++ to create SHM
        print("[Supervisor] Waiting for Executive READY signal...")

        # Open pipe for reading (blocking open until C++ opens for writing)
        self.report_fd = os.open(REPORT_PIPE_PATH, os.O_RDONLY | os.O_NONBLOCK)

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

        print("[Supervisor] Executive is READY. Attaching to Shared Memory...")
        try:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME)
            # Create a NumPy view of the raw RAM (Zero-Copy)
            self.tensor_view = np.ndarray((4096,), dtype=np.float32, buffer=self.shm.buf)
            print("[Supervisor] Shared Memory Attached.")
        except FileNotFoundError:
             print(f"[Error] Shared memory {SHM_NAME} not found. C++ failed to create it?")
             self.cleanup()
             sys.exit(1)

        # Open Command Pipe
        self.command_fd = os.open(COMMAND_PIPE_PATH, os.O_WRONLY)

    def load_gpu_shard(self):
        print("[Supervisor] Loading GPU Shard (PyTorch)...")
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

        if os.path.exists(REPORT_PIPE_PATH): os.remove(REPORT_PIPE_PATH)
        if os.path.exists(COMMAND_PIPE_PATH): os.remove(COMMAND_PIPE_PATH)

        print("[Cleanup] Done.")

if __name__ == "__main__":
    supervisor = OfferingSupervisor()
    atexit.register(supervisor.cleanup)

    supervisor.setup_resources()
    supervisor.launch_executive()
    supervisor.load_gpu_shard()
    supervisor.inference_loop()
