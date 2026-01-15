# Super-Intel-NPU

A high-performance hybrid inference engine for Intel Core Ultra NPU, designed for low-latency, "Zero-Copy" AI applications.

## Project Structure

*   `src/python/`: Python Supervisor, Bake scripts, and logic.
*   `src/cpp/`: C++ Executive for Shared Memory management.
*   `models/`: Storage for baked OpenVINO artifacts.
*   `scripts/`: Utility scripts for setup.

## Quick Start (New Offering)

1.  **Environment Setup:**
    ```bash
    ./scripts/setup_env.sh
    ```

2.  **Bake the Model:**
    Downloads and converts the model to NPU-optimized INT4.
    ```bash
    python src/python/bake_model.py
    ```

3.  **Run Inference (Recommended):**
    Use the master launch script to ensure hardware health and proper logging.
    ```bash
    ./launch.sh
    ```

    *Alternatively (Advanced):*
    ```bash
    python src/python/supervisor.py --prompt "Explain quantum computing"
    ```

## Diagnostic Tools

The project includes a suite of tools to prevent NPU hangs:

*   **`./launch.sh`**: The master switch. Handles Hard Reset -> Vitals Check -> Execution.
*   **`scripts/npu_reset.sh`**: Hard resets the Linux kernel NPU driver (use if device disappears).
*   **`src/python/preflight_check.py`**: Audits the model binary to ensure it fits in NPU memory (INT4 verification).
*   **`src/python/npu_vitals.py`**: checks Thermal and RAM status.

## Documentation
See the `documentation/` folder for detailed technical logs of architecture decisions and fixes.
