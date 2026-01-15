#!/bin/bash

# --- CONFIGURATION ---
MODEL_XML="./models/neuralchat_int4/openvino_model.xml"
TOKENIZER_ID="Intel/neural-chat-7b-v3-1"
CACHE_DIR="./model_cache"
TEST_PROMPT="Note the total number of planets; then List those planets starting from Mercury."
LOG_DIR="./log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
HISTORY_LOG="$LOG_DIR/npu_history_$TIMESTAMP.log.txt"
PERSISTENT_LOG="npu_history.log"

# --- LOGGING HEADER ---
echo "----------------------------------------------------" | tee -a $HISTORY_LOG $PERSISTENT_LOG
echo "START SESSION: $TIMESTAMP" | tee -a $HISTORY_LOG $PERSISTENT_LOG

echo "====================================================" | tee -a $HISTORY_LOG
echo "      NEW OFFERING: NPU MASTER LAUNCH SEQUENCE      " | tee -a $HISTORY_LOG
echo "====================================================" | tee -a $HISTORY_LOG

# 1. HARD RESET
echo "[1/5] Performing NPU Hard Reset..." | tee -a $HISTORY_LOG
# Execute reset script if it exists
if [ -f "scripts/npu_reset.sh" ]; then
    bash scripts/npu_reset.sh >> $HISTORY_LOG 2>&1
else
    echo "Warning: scripts/npu_reset.sh not found." | tee -a $HISTORY_LOG
fi
echo "[SUCCESS] Reset Complete." | tee -a $HISTORY_LOG

# 2. PURGE CACHE
echo "[2/5] Purging Model Cache..." | tee -a $HISTORY_LOG
rm -rf ${CACHE_DIR}/*

# 3. VITALS CHECK
echo "[3/5] Auditing System Resources..." | tee -a $HISTORY_LOG
python3 src/python/npu_vitals.py >> $HISTORY_LOG 2>&1
if [ $? -ne 0 ]; then
    echo "!! VITALS CHECK FAILED. Proceeding with caution..." | tee -a $HISTORY_LOG
fi

# 4. PRE-FLIGHT AUDIT
echo "[4/5] Running Model Pre-Flight Audit..." | tee -a $HISTORY_LOG
python3 src/python/preflight_check.py "$MODEL_XML" >> $HISTORY_LOG 2>&1
if [ $? -ne 0 ]; then
    echo "!! PRE-FLIGHT AUDIT FAILED. Check logs." | tee -a $HISTORY_LOG
fi

# 5. EXECUTION (Capture output to log and screen)
echo "[5/5] Launching Supervisor..." | tee -a $HISTORY_LOG
echo "----------------------------------------------------" | tee -a $HISTORY_LOG
# We use 'tee' to show output on screen AND save it to the log
python3 src/python/supervisor.py \
    --model_xml "$MODEL_XML" \
    --tokenizer_id "$TOKENIZER_ID" \
    --prompt "$TEST_PROMPT" \
    --max_tokens 128 | tee -a $HISTORY_LOG

# --- DIAGNOSTIC FOOTER ---
{
    echo -e "\n--- DIAGNOSTICS ---"
    echo "Device Node: $(ls /dev/accel/accel0 2>/dev/null)"
    echo "OpenVINO: $(python3 -c 'import openvino; print(openvino.__version__)' 2>/dev/null)"
    echo "Driver: $(dpkg -l | grep intel-level-zero-npu | awk '{print $3}' 2>/dev/null)"
    echo "Kernel VPU Status:"
    sudo dmesg | grep -i vpu | tail -n 5
    echo "END SESSION"
    echo "----------------------------------------------------"
} >> $HISTORY_LOG

# Duplicate footer to persistent log
tail -n 10 $HISTORY_LOG >> $PERSISTENT_LOG

echo -e "\n[FINISH] Run complete. Diagnostics saved to $HISTORY_LOG"
