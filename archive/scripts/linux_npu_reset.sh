#!/bin/bash
# NPU Hard Reset Script for Intel Core Ultra
echo "[System] Initiating NPU Driver Reset..."

# 1. Kill any processes using the NPU
sudo pkill -9 -f supervisor.py
sudo pkill -9 -f executive_shard

# 2. Unload the VPU driver (NPU)
# This clears hardware locks and resets the MMU
sudo modprobe -r intel_vpu

# 3. Reload the driver
sudo modprobe intel_vpu

# 4. Verify visibility
if ls /dev/accel/accel0 > /dev/null 2>&1; then
    echo "[SUCCESS] NPU driver reloaded and /dev/accel/accel0 is visible."
else
    echo "[ERROR] NPU failed to initialize. Check 'sudo dmesg | tail -n 20'."
fi
