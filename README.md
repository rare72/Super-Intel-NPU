# New Offering: High-Performance Hybrid Inference Engine

This project provides a standalone, high-performance hybrid inference engine optimized for Intel Core Ultra processors (Meteor Lake / Arrow Lake). It utilizes a split-architecture approach where the heavy "prefill" and initial layers are processed on the Intel NPU, and the final generation steps are handled by the iGPU or dGPU.

## Architecture

The system consists of two main components:

1.  **C++ Executive (`executive_shard`):** A persistent, low-level binary that manages hardware discovery (Level Zero), allocates Shared Memory (POSIX SHM), and monitors system health.
2.  **Python Supervisor (`supervisor.py`):** The user-facing controller that manages the model "Bake" process, handles tokenization, and orchestrates the inference loop via zero-copy memory handoffs.

## Prerequisites

*   **OS:** Linux (tested on Ubuntu 24.04 / Xubuntu)
*   **Hardware:** Intel Core Ultra Processor (with NPU and Arc Graphics)
*   **Drivers:** Intel Level Zero (`intel-level-zero-npu`, `intel-level-zero-gpu`), OpenVINO 2025.0+.

## Installation

1.  **Set up the environment:**
    ```bash
    bash scripts/setup_env.sh
    source venv_offering/bin/activate
    ```

2.  **Compile the Executive:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    cd ..
    ```

## Usage

### 1. Bake the Model (One-Time Setup)
Before running inference, you must "bake" the model. This downloads the weights, converts them to OpenVINO IR, compresses them to INT4, and enforces strict static shapes for NPU compatibility.

```bash
python3 src/python/bake_model.py --model_id Intel/neural-chat-7b-v3-1
```
*   **Output:** Optimized model binaries in `./offering_int4_binary` (or specified output).
*   **Note:** This process takes several minutes. Ensure you have ~30GB of free space for the conversion.

### 2. Run Inference
Launch the Supervisor to start the interactive inference session.

```bash
python3 src/python/supervisor.py
```

## Troubleshooting

*   **Bus Error / Segmentation Fault:** Often caused by dynamic shapes on the NPU. Ensure you re-run `bake_model.py` if you change model configurations.
*   **"Node beam_idx is still dynamic":** Fixed in `bake_model.py` by enforcing `[1]` static shape.
*   **Permissions:** Ensure your user is in the `render` group (`sudo usermod -a -G render $USER`) and re-login.

## Directory Structure
*   `src/python/`: Python source code (Baker, Supervisor).
*   `src/cpp/`: C++ source code (Executive).
*   `scripts/`: Setup and utility scripts.
*   `documentation/`: Change logs and technical notes.
