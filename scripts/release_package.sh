#!/bin/bash
# scripts/release_package.sh
# Bundles the New Offering into a distributable folder.

RELEASE_DIR="release_v1.0"
MODEL_SRC="./models/neuralchat_int4"
BINARY_SRC="./src/cpp/build/executive_shard"
PYTHON_SRC="./src/python/supervisor.py"

echo ">>> [Release] Packaging New Offering..."

# 1. Create Directory Structure
if [ -d "$RELEASE_DIR" ]; then
    echo "    Cleaning old release directory..."
    rm -rf "$RELEASE_DIR"
fi

mkdir -p "$RELEASE_DIR/models"
mkdir -p "$RELEASE_DIR/bin"

# 2. Copy Artifacts
echo "    Copying Python Supervisor..."
cp "$PYTHON_SRC" "$RELEASE_DIR/supervisor.py"

if [ -d "$MODEL_SRC" ]; then
    echo "    Copying Model Artifacts from $MODEL_SRC..."
    cp -r "$MODEL_SRC" "$RELEASE_DIR/models/"
else
    echo "    [Warning] Model directory $MODEL_SRC not found. Release will be empty of weights."
fi

if [ -f "$BINARY_SRC" ]; then
    echo "    Copying C++ Executive..."
    cp "$BINARY_SRC" "$RELEASE_DIR/bin/executive_shard"
else
    echo "    [Warning] C++ Binary not found at $BINARY_SRC. Skipping."
fi

# 3. Generate Runner Script
echo "    Generating run.sh..."
cat <<EOF > "$RELEASE_DIR/run.sh"
#!/bin/bash
# Launcher for New Offering

# Ensure we are in the script directory
cd "\$(dirname "\$0")"

# Define Paths
MODEL_DIR="./models/neuralchat_int4"

# Check if model exists
if [ ! -d "\$MODEL_DIR" ]; then
    echo "Error: Model directory not found at \$MODEL_DIR"
    exit 1
fi

echo "Starting New Offering Supervisor..."
python3 supervisor.py --model_xml "\$MODEL_DIR" --device NPU "\$@"
EOF

chmod +x "$RELEASE_DIR/run.sh"

# 4. Generate README
echo "    Generating README.md..."
cat <<EOF > "$RELEASE_DIR/README.md"
# New Offering: High-Performance NPU Inference

## Overview
This package contains the optimized "New Offering" inference engine for Intel Core Ultra NPU.
It utilizes a hybrid Python/C++ architecture with Zero-Copy Shared Memory handoffs.

## Requirements
- Intel Core Ultra Processor (Series 1 or 2)
- Linux Kernel 6.8+ (for NPU driver support)
- Python 3.10+
- Intel Level Zero Drivers

## Installation
1. Ensure you have the Intel NPU drivers installed:
   \`sudo apt install intel-level-zero-npu intel-level-zero-gpu\`

2. Install Python dependencies:
   \`pip install numpy torch openvino optimum-intel transformers\`

## Usage
To run the interactive chat:
\`./run.sh\`

To run a single prompt:
\`./run.sh --prompt "Why is the sky blue?"\`

## Troubleshooting
- If you see **ZE_RESULT_ERROR_UNKNOWN**, ensure no other NPU processes are running.
- Use \`./run.sh --device CPU\` to fallback if the NPU is unstable.
EOF

echo ">>> [Release] Package created successfully at $RELEASE_DIR"
