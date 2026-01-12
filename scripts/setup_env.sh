#!/bin/bash
set -e

echo ">>> Starting New Offering Environment Setup..."

# 1. System Drivers & Tools (Intel Core Ultra Support)
# Checks if running as root or with sudo capability for system packages
if [ "$EUID" -ne 0 ]; then
  echo ">>> Note: You are running as a non-root user. You may be prompted for sudo password for system updates."
  SUDO="sudo"
else
  SUDO=""
fi

echo ">>> Updating apt repositories..."
$SUDO apt update

echo ">>> Installing Intel Level Zero, OpenCL, and Build Tools..."
# Using DEBIAN_FRONTEND=noninteractive to avoid prompts in scripts
$SUDO DEBIAN_FRONTEND=noninteractive apt install -y \
    libze1 \
    libze-dev \
    intel-level-zero-gpu \
    intel-opencl-icd \
    clinfo \
    cmake \
    build-essential \
    python3-venv \
    python3-dev

# Note: For NPU support on Xubuntu 24.04+, ensure the NPU driver is installed
# It might be named differently depending on the repo, but often included in modern intel stacks.
# Attempting to install specific NPU level zero driver if available in repo
echo ">>> Attempting to install NPU specific driver (if available)..."
$SUDO DEBIAN_FRONTEND=noninteractive apt install -y intel-level-zero-npu || echo ">>> Warning: intel-level-zero-npu package not found. Ensure your repo has NPU drivers."

# 2. User Groups
echo ">>> Adding user to 'render' group for hardware access..."
$SUDO usermod -a -G render $USER || echo ">>> Warning: Could not add user to render group."

# 3. Python Environment
VENV_NAME="venv_offering"
echo ">>> Setting up Python Virtual Environment: $VENV_NAME"

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
fi

# Activate venv
source $VENV_NAME/bin/activate

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing Python Dependencies..."
# Installing core AI and Intel optimization libraries
pip install "optimum-intel[openvino,nncf]" \
    "openvino-dev>=2025.0.0" \
    "huggingface-hub" \
    "torch" \
    "setuptools" \
    "numpy"

# 4. Verification
echo ">>> Verifying Installation..."

echo "--- CMake Version ---"
cmake --version

echo "--- Python Version ---"
python --version

echo "--- OpenVINO Package ---"
python -c "import openvino; print(f'OpenVINO version: {openvino.runtime.get_version()}')"

echo ">>> Setup Complete. Please REBOOT or log out/in for group changes to take effect."
echo ">>> To activate the environment later, run: source $VENV_NAME/bin/activate"
