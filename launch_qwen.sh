#!/bin/bash
# Master Launch Script for Qwen3-8B (INT4/NPU)

MODEL_DIR="./models/qwen3_int4"
LOG_FILE="qwen_history.log"

echo "=== Qwen3 Launch Sequence ===" | tee -a $LOG_FILE
date | tee -a $LOG_FILE

# 1. Reset NPU
echo "[1/4] Resetting NPU..."
sudo bash scripts/npu_reset.sh >> $LOG_FILE 2>&1

# 2. Vitals
echo "[2/4] Checking Vitals..."
python3 src/python/npu_vitals.py >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then echo "!! Low Resources !!"; fi

# 3. Pre-flight (Custom len)
echo "[3/4] Pre-flight Check..."
# Note: We need to update preflight to ignore size > 4GB slightly as Qwen might be larger, or trust bake
python3 src/python/preflight_check.py "$MODEL_DIR/openvino_model.xml" >> $LOG_FILE 2>&1

# 4. Run
echo "[4/4] Starting Inference..."
python3 src/python/supervisor_qwen.py --model_dir "$MODEL_DIR" --prompt "Explain the concept of recursion."
