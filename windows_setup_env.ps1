# windows_setup_env.ps1
# Setup script for Super-Intel-NPU on Windows 11

Write-Host "--- Super-Intel-NPU Windows Setup ---" -ForegroundColor Cyan

# 1. Check Python
$pythonVersion = python --version
if ($?) {
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Error "Python not found! Please install Python 3.12+ and add to PATH."
    exit 1
}

# 2. Create Virtual Environment
if (-not (Test-Path "openvino_env")) {
    Write-Host "Creating virtual environment 'openvino_env'..."
    python -m venv openvino_env
} else {
    Write-Host "Virtual environment 'openvino_env' already exists."
}

# 3. Activate and Install
Write-Host "Activating environment and installing dependencies..."
# Note: We cannot easily change the shell context of the running script to the venv for subsequent commands
# in the same way 'source' does in bash. We will call pip directly via the venv python executable.

$pipPath = ".\openvino_env\Scripts\pip.exe"

& $pipPath install --upgrade pip
& $pipPath install openvino==2025.4.1 openvino-genai==2025.4.1 huggingface-hub optimum-intel nncf

Write-Host "--- Setup Complete ---" -ForegroundColor Cyan
Write-Host "To activate manually, run: .\openvino_env\Scripts\Activate.ps1"
