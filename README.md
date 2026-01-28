# ⚠️ STATUS: PLATFORM SWITCH IN PROGRESS (Linux -> Windows) ⚠️

**NOTE:** This repository is currently migrating from a Linux-based environment to **Windows 11 (24H2)**.
Legacy Linux scripts have been moved to the `archive/` directory.

The previous "NOT WORKING" status regarding the Intel Ultra 5 225 (Rev 10) on Linux stands, but the hardware is reportedly stable on Windows 11 using the new toolchain.

---

# Super-Intel-NPU (Windows 11 Edition)

A high-performance hybrid inference engine for Intel Core Ultra NPU, designed for low-latency, "Zero-Copy" AI applications.

## Windows 11 Migration (Arrow Lake NPU)

We have pivoted to **Windows 11 (24H2)** using the **Visual Studio 2026 Build Tools (v18.2.1)**.

*   **Compiler:** MSVC v145 (Native optimizations for Arrow Lake NPU 3).
*   **Build System:** VS Build Tools (replaces standalone Make/CMake).
*   **Hardware:** Intel Ultra 5 225 (Rev 10) is fully supported via the MCDM driver model.

### Environment Isolation Strategy
To ensure the NPU (Intel AI Boost) is correctly isolated from NVIDIA GPUs and utilizes the full 31.8GB aperture:
*   `OV_NPU_LOG_LEVEL = LOG_INFO`
*   `ZE_AFFINITY_MASK = 0.0` (Isolates NPU 0)
*   `ZE_ENABLE_PCI_ID_DEVICE_ORDER = 1`

## Project Structure

*   `src/python/`: Python Supervisor, Bake scripts, and logic.
*   `src/cpp/`: C++ Executive (Source).
*   `models/`: Storage for baked OpenVINO artifacts.
*   `archive/`: Legacy Linux scripts.

## Quick Start (Windows)

1.  **Environment Setup:**
    Open PowerShell as Administrator:
    ```powershell
    .\windows_setup_env.ps1
    ```

2.  **Bake the Model (Phi-4):**
    Downloads `microsoft/phi-4-onnx` (GPU Branch) and converts to NPU-Optimized INT4.
    ```powershell
    python src/python/bake_phi4.py
    ```

3.  **Run Inference:**
    Use the Windows launch script to set environment variables and start the supervisor.
    ```powershell
    .\windows_launch.ps1
    ```

## Diagnostic Tools

*   **`src/python/verify_npu_isolation.py`**: Verifies that the NPU is isolated and the NVIDIA GPU is hidden from OpenVINO.
*   **`src/python/npu_vitals.py`**: Checks thermal and RAM status.

## Documentation
See the `documentation/` folder for technical logs, including the `Hardware_Failure_Report` for the Linux context.
