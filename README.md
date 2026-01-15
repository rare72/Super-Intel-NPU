# Super-Intel-NPU

A high-performance hybrid inference engine for Intel Core Ultra NPU, designed for low-latency, "Zero-Copy" AI applications.

## Project Structure

*   `src/python/`: Python Supervisor, Bake scripts, and logic.
*   `src/cpp/`: C++ Executive for Shared Memory management.
*   `models/`: Storage for baked OpenVINO artifacts.
*   `scripts/`: Utility scripts for setup and release.
*   `release_v1.0/`: **(New)** Distributable package containing the runtime.

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

3.  **Run Inference:**
    ```bash
    python src/python/supervisor.py --prompt "Explain quantum computing"
    ```

4.  **Create Release Package:**
    Bundles the application for distribution.
    ```bash
    ./scripts/release_package.sh
    ```

## Documentation
See the `documentation/` folder for detailed technical logs of architecture decisions and fixes.
