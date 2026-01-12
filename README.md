# New Offering: Intel Core Ultra Hybrid Inference

## Overview
This project implements a high-performance, hardware-sharded inference engine designed for Intel Core Ultra (Meteor Lake / Arrow Lake) processors. It splits Large Language Models (LLMs) between the NPU (Neural Processing Unit) and GPU (Arc Graphics) to maximize throughput and efficiency.

## Key Features
- **Data-Free INT4 Compression**: Uses NNCF (Neural Network Compression Framework) to compress weights to 4-bit without requiring calibration data.
- **Hardware Sharding**:
  - **Shard A (NPU)**: Runs the first half of the model using OpenVINO.
  - **Shard B (GPU)**: Runs the second half using PyTorch/CUDA (or Arc equivalent).
- **Zero-Copy Handoff**: Utilizes POSIX Shared Memory to transfer tensor data between the C++ Executive and Python Supervisor without copying memory.
- **Robust Lifecycle**: Python Supervisor manages the C++ process, ensuring clean startup and shutdown (no memory leaks).

## Directory Structure
- `src/cpp`: C++ Source code for the low-level Executive process (Level Zero, OpenVINO).
- `src/python`: Python scripts for model baking, supervision, and inference.
- `scripts`: Helper scripts for setup, hardware checking, and packaging.
- `documentation`: Implementation details and logs.

## Prerequisites
- **Hardware**: Intel Core Ultra Processor (Series 1 or 2) with NPU and Arc Graphics.
- **OS**: Linux (Ubuntu 22.04 / 24.04 or compatible).
- **Drivers**: Intel Level Zero GPU & NPU drivers (`intel-level-zero-gpu`, `intel-level-zero-npu`).

## Quick Start

### 1. Environment Setup
Run the setup script to install system dependencies and create the Python virtual environment.
```bash
./scripts/setup_env.sh
source venv_offering/bin/activate
```

### 2. Verify Hardware
Check if your NPU is visible to the drivers.
```bash
python scripts/check_intel_hw.py
```

### 3. "Bake" the Model
Download and compress the model (e.g., Llama-3-8B). This runs NNCF INT4 compression.
```bash
python src/python/bake_model.py --model_id meta-llama/Meta-Llama-3-8B
```

### 4. Compile the Executive
Build the C++ binary that manages the NPU.
```bash
mkdir -p src/cpp/build
cd src/cpp/build
cmake ..
make
cd ../../..
```

### 5. Run the Offering
Launch the Supervisor. It will start the C++ Executive and enter the inference loop.
```bash
python src/python/supervisor.py
```

## Troubleshooting
- **CMake Error "Level Zero not found"**: Ensure `libze-dev` is installed.
- **Shared Memory Errors**: Ensure `/dev/shm` is mounted and you have permissions (standard on most Linux distros).
- **NPU not found**: Verify kernel version (6.8+ recommended) and `intel-level-zero-npu` package.
