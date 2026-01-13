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
- **Detailed Metrics**: Reports precise Token-Per-Second (TPS) statistics for NPU, C++ Framework, and GPU execution.

## Supported Models
This engine is optimized for the following models:
1.  **Meta Llama 3 (8B)**: The standard open-weight model.
2.  **Intel Neural Chat 7B (v3-1)**: A Mistral-based model fine-tuned on Gaudi2, utilizing SwiGLU activation for native NPU support.

## Prerequisites
- **Hardware**: Intel Core Ultra Processor (Series 1 or 2) with NPU and Arc Graphics.
- **OS**: Linux (Ubuntu 24.04 Noble Numbat recommended).
- **Drivers**: Intel Level Zero GPU & NPU drivers (installed via setup script).

## Quick Start

### 1. Environment Setup
Run the setup script. This script **adds the Intel Graphics Package Repository** and installs necessary system drivers (`libze`, `opencl`, etc.).

**Note for Ubuntu 24.04 Users:** The script automatically handles the modern package naming (`libegl1` vs `libegl1-mesa`).

```bash
./scripts/setup_env.sh
source venv_offering/bin/activate
```

### 2. Verify Hardware
Check if your NPU is visible to the drivers.
```bash
python scripts/verify_install.py
```
Or use the detailed check:
```bash
python scripts/check_intel_hw.py
```

### 3. "Bake" the Model
You must download and compress the model before running it.

**IMPORTANT:** If you encounter `ZE_RESULT_ERROR_INVALID_ARGUMENT` or shape errors, you MUST delete the output directory and re-run this command.

**Option A: Meta Llama 3 (8B)**
```bash
rm -rf ./models/llama3_int4
python src/python/bake_model.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --output_dir ./models/llama3_int4
```

**Option B: Intel Neural Chat 7B (v3-1)**
```bash
rm -rf ./models/neuralchat_int4
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
Launch the Supervisor to generate text. You can run in interactive mode or single-shot mode.

**Interactive Mode:**
```bash
python src/python/supervisor.py \
    --model_xml ./models/neuralchat_int4 \
    --tokenizer_id Intel/neural-chat-7b-v3-1
```

**Single-Shot Prompting (Real Inference):**
```bash
python src/python/supervisor.py \
    --model_xml ./models/neuralchat_int4 \
    --tokenizer_id Intel/neural-chat-7b-v3-1 \
    --prompt "What are the benefits of NPU inference?"
```

## Performance Metrics
The supervisor now outputs detailed execution statistics on exit:
```text
[EXIT] Script finished at Mon Jan 12 ...
[EXIT] Total Input Prompt Tokens: 15
[EXIT] NPU_Duration_Time:  3.4000
[EXIT] C++ FrameWork-(Hand Off Processing)_Duration Time: 0.015
[EXIT] GPU_Duration_Time: 1.6000
[EXIT] Total Context Tokens (Prompt + Generated): 143
[EXIT] Total Readable Tokens (Answer content): 128
[EXIT] Tokens per Second_NPU: 37.65
...
```

## Troubleshooting

### Setup Script Fails on Ubuntu 24.04
If you see errors about `libegl1-mesa` or `intel-level-zero-gpu` not being found:
1.  Ensure you have internet access so the Intel GPG key can be downloaded.
2.  The script has been updated to use `libegl1` (the modern package name). Rerun `./scripts/setup_env.sh`.
3.  If issues persist, verify you have the `universe` and `multiverse` repositories enabled in Ubuntu.

### NPU Not Visible
If `verify_install.py` does not list 'NPU':
1.  Verify you are in the `render` group: `groups $USER`.
2.  Check `dmesg | grep intel_vpu` to see if the kernel driver loaded.
3.  Reboot your system after running the setup script.

### ZE_RESULT_ERROR_INVALID_ARGUMENT / Dynamic Shape Error
If the supervisor logs `Upper bounds are not specified` or `to_shape was called on a dynamic shape`:
1.  This means the model on disk does not have NPU-compatible shapes.
2.  **Fix:** Delete your baked model directory (e.g., `rm -rf ./models/neuralchat_int4`) and re-run `bake_model.py`. The new script enforces bounded shapes.
