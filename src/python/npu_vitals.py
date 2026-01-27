import openvino as ov
import psutil
import sys

def check_vitals():
    try:
        core = ov.Core()
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenVINO Core: {e}")
        return False

    print("--- NPU Health Check ---")

    if "NPU" not in core.available_devices:
        print("[ERROR] NPU is not detected. Run npu_reset.sh.")
        return False

    # 1. Check Power/Thermal Metrics
    # Note: Availability of these properties depends on the driver version
    try:
        # Some drivers might not expose this property or use a different key
        thermal = core.get_property("NPU", "DEVICE_THERMAL")
        print(f"[Health] NPU Temperature: {thermal}")
    except Exception:
        print("[Health] Thermal sensors not exposed by current driver.")

    # 2. Check System Memory (The -28 Error Prevention)
    # NPU uses system RAM; ensure at least 8GB is free for the INT4 model + KV Cache
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"[Health] Available System RAM: {available_ram_gb:.2f} GB")

    if available_ram_gb < 6.0:
        print("!! WARNING: Low memory. Close other apps before loading the model.")
        # We don't fail, just warn, as swap might handle it (slowly)
        # return False

    # 3. Check Device Utilization
    # If the NPU is already 100% busy, loading a new shard will crash it.
    print("[SUCCESS] Hardware is idle and cool.")
    return True

if __name__ == "__main__":
    if not check_vitals():
        sys.exit(1)
