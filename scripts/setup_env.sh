#!/bin/bash
set -e

echo ">>> Starting New Offering Environment Setup..."

# 1. System Drivers & Tools (Intel Core Ultra Support)
if [ "$EUID" -ne 0 ]; then
  echo ">>> Note: You are running as a non-root user. You may be prompted for sudo password for system updates."
  SUDO="sudo"
else
  SUDO=""
fi

echo ">>> Installing Prerequisites (wget, gpg)..."
$SUDO apt update
$SUDO apt install -y wget gpg software-properties-common

# 2. Add Intel Graphics Repository (Required for Core Ultra / Level Zero)
echo ">>> Adding Intel Graphics Repository..."
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  $SUDO gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Detect Ubuntu Codename (e.g., noble, jammy)
if command -v lsb_release &> /dev/null; then
    CODENAME=$(lsb_release -cs)
else
    # Fallback/Default to Noble (24.04) as this is the target OS
    CODENAME="noble"
fi

echo ">>> Detected Ubuntu Codename: $CODENAME"

# Add the repository (Client Repo for Core Ultra)
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${CODENAME} client" | \
  $SUDO tee /etc/apt/sources.list.d/intel-gpu-${CODENAME}.list

echo ">>> Updating apt repositories with Intel source..."
$SUDO apt update

echo ">>> Installing Intel Level Zero, OpenCL, and Build Tools..."
# Ubuntu 24.04 (Noble) Specific Fixes:
# - Replaced legacy 'libegl1-mesa' with 'libegl1'
# - Added 'libtbb12' and 'libgl1-mesa-dev' explicitly
# - Using DEBIAN_FRONTEND=noninteractive to avoid prompts

$SUDO DEBIAN_FRONTEND=noninteractive apt install -y \
    libegl1 \
    libgl1-mesa-dev \
    libgbm1 \
    libtbb12 \
    libze1 \
    libze-dev \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-media-va-driver-non-free \
    libmfxgen1 \
    libvpl2 \
    libegl1-mesa-dev \
    libgl1-mesa-dri \
    libglapi-mesa \
    libgles2-mesa-dev \
    libglx-mesa0 \
    libigdgmm12 \
    libxatracker2 \
    mesa-va-drivers \
    mesa-vdpau-drivers \
    mesa-vulkan-drivers \
    va-driver-all \
    vainfo \
    hwinfo \
    clinfo \
    cmake \
    build-essential \
    python3-venv \
    python3-dev \
    python3-pip

# Attempt NPU specific driver if not covered by above
# Note: In some Intel repos, this package might be named differently or included in level-zero-gpu
$SUDO DEBIAN_FRONTEND=noninteractive apt install -y intel-level-zero-npu || echo ">>> Note: intel-level-zero-npu package not explicitly found (might be included in main packages or manually installed)."

# 3. User Groups
echo ">>> Adding user to 'render' group for hardware access..."
$SUDO usermod -a -G render $USER || echo ">>> Warning: Could not add user to render group."

# 4. Python Environment
# Determine directory relative to script or current location
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
VENV_PATH="$PROJECT_ROOT/venv_offering"

echo ">>> Setting up Python Virtual Environment at: $VENV_PATH"

# Always recreate or ensure venv exists
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
else
    echo ">>> Virtual environment already exists. Updating..."
fi

# Activate venv for this script execution
source "$VENV_PATH/bin/activate"

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing Python Dependencies..."
pip install "optimum-intel[openvino,nncf]" \
    "openvino>=2025.0.0" \
    "huggingface-hub" \
    "torch" \
    "setuptools" \
    "numpy"

# 5. Verification
echo ">>> Verifying Installation..."
cmake --version
python --version
python -c "import openvino; print(f'OpenVINO version: {openvino.runtime.get_version()}')"

echo ">>> Setup Complete!"
echo ">>> IMPORTANT: Please REBOOT your machine for driver changes to take effect."
echo ">>> To start working, run this command manually:"
echo "    source venv_offering/bin/activate"
