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

### 1. Bake the Model (Mandatory One-Time Setup)
For the "New Offering" targeting the Intel NPU, you **must** bake the model. Standard Hugging Face downloads will likely fail on the NPU due to dynamic shape drivers issues.

The Bake process:
1.  **Downloads** the open weights.
2.  **Compresses** weights to INT4 (fitting 28GB models into ~5GB RAM).
3.  **Reshapes** the graph to strict static inputs (preventing driver crashes).
4.  **Packages** the necessary config files for the Supervisor.

```bash
# Example: Bake Neural Chat to a local folder
python3 src/python/bake_model.py \
    --model_id Intel/neural-chat-7b-v3-1 \
    --output_dir ./models/neuralchat_int4
```
*   **Output:** Optimized model binaries (`.xml`, `.bin`) and configuration files (`.json`) in the output directory.
*   **Note:** This process takes several minutes. Ensure you have ~30GB of free space for the conversion.

### 2. Run Inference
Launch the Supervisor pointing to your **baked** model directory.

```bash
python3 src/python/supervisor.py \
    --model_xml ./models/neuralchat_int4 \
    --tokenizer_id Intel/neural-chat-7b-v3-1 \
    --prompt "Explain quantum computing."
```
*   **Flags:**
    *   `--model_xml`: Path to the baked model directory (or the .xml file inside it).
    *   `--device`: Target device (default: `NPU`).
    *   `--chat_style`: Template format (`neural`, `llama3`, or `raw`).

## Troubleshooting

*   **"The library name could not be automatically inferred":**
    *   **Cause:** The model directory contains OpenVINO files (`.xml`) but is missing the Hugging Face `config.json`.
    *   **Fix:** Re-run `bake_model.py` (updated Jan 14, 2026) which now correctly copies these files to the output folder.
*   **Bus Error / Segmentation Fault:**
    *   **Cause:** Dynamic shapes on older NPU drivers.
    *   **Fix:** Ensure you are using a baked model with strict static shapes (`--use_cache=False` in `bake_model.py`).
*   **"Node beam_idx is still dynamic":**
    *   **Fix:** Fixed in `bake_model.py` by enforcing `[1]` static shape.
*   **NPU Not Found:**
    *   **Fix:** Ensure your user is in the `render` group (`sudo usermod -a -G render $USER`) and you have installed `intel-level-zero-npu`. Verify with `python3 -c "import openvino as ov; print(ov.Core().available_devices)"`.

## Directory Structure
*   `src/python/`: Python source code (Baker, Supervisor).
*   `src/cpp/`: C++ source code (Executive).
*   `scripts/`: Setup and utility scripts.
*   `documentation/`: Change logs and technical notes.
