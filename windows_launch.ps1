# windows_launch.ps1
# Master Launch Script for Super-Intel-NPU (Windows 11)

Write-Host "--- Super-Intel-NPU Windows Launch ---" -ForegroundColor Cyan

# 1. Set Environment Variables for NPU Isolation
$env:OV_NPU_LOG_LEVEL = "LOG_INFO"
$env:ZE_AFFINITY_MASK = "0.0"
$env:ZE_ENABLE_PCI_ID_DEVICE_ORDER = "1"
$env:OPENVINO_LOG_LEVEL = "1"

Write-Host "Environment Variables Set:"
Write-Host "  OV_NPU_LOG_LEVEL: $env:OV_NPU_LOG_LEVEL"
Write-Host "  ZE_AFFINITY_MASK: $env:ZE_AFFINITY_MASK"
Write-Host "  ZE_ENABLE_PCI_ID_DEVICE_ORDER: $env:ZE_ENABLE_PCI_ID_DEVICE_ORDER"

# 2. Verify Isolation (Optional but Recommended)
$pythonPath = ".\openvino_env\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    # Fallback to system python if venv not found
    $pythonPath = "python"
}

Write-Host "`n[1/3] Verifying NPU Isolation..."
& $pythonPath src/python/verify_npu_isolation.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Isolation check failed. Proceeding with caution."
}

# 3. Vitals Check
Write-Host "`n[2/3] Checking Vitals..."
& $pythonPath src/python/npu_vitals.py

# 4. Launch Supervisor
Write-Host "`n[3/3] Launching Supervisor..."
Write-Host "----------------------------------------------------"
# Pass all arguments from script to python script
& $pythonPath src/python/supervisor.py @args
