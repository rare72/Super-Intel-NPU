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

## Supported Models
This engine is optimized for the following models:
1.  **Meta Llama 3 (8B)**: The standard open-weight model.
2.  **Intel Neural Chat 7B (v3-1)**: A Mistral-based model fine-tuned on Gaudi2, utilizing SwiGLU activation for native NPU support.

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
You must download and compress the model before running it.

**Option A: Meta Llama 3 (8B)**
```bash
python src/python/bake_model.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --output_dir ./models/llama3_int4
```

**Option B: Intel Neural Chat 7B (v3-1)**
```bash
python src/python/bake_model.py \
    --model_id Intel/neural-chat-7b-v3-1 \
    --output_dir ./models/neuralchat_int4
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
Launch the Supervisor. Specify which baked model to use.

**Running Llama 3:**
```bash
python src/python/supervisor.py \
    --model_xml ./models/llama3_int4/openvino_model.xml \
    --tokenizer_id meta-llama/Meta-Llama-3-8B
```

**Running Neural Chat:**
```bash
python src/python/supervisor.py \
    --model_xml ./models/neuralchat_int4/openvino_model.xml \
    --tokenizer_id Intel/neural-chat-7b-v3-1
```

## Troubleshooting
- **CMake Error "Level Zero not found"**: Ensure `libze-dev` is installed.
- **Shared Memory Errors**: Ensure `/dev/shm` is mounted and you have permissions (standard on most Linux distros).
- **NPU not found**: Verify kernel version (6.8+ recommended) and `intel-level-zero-npu` package.
