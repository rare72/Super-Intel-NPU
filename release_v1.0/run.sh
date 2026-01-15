#!/bin/bash
# Launcher for New Offering

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Define Paths
MODEL_DIR="./models/neuralchat_int4"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found at $MODEL_DIR"
    exit 1
fi

echo "Starting New Offering Supervisor..."
python3 supervisor.py --model_xml "$MODEL_DIR" --device NPU "$@"
