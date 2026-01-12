import openvino as ov
import os
import subprocess

def check_npu_visibility():
    print(">>> Intel Core Ultra Hardware Sanity Check <<<")

    # 1. Check for Level Zero Loader in System Paths
    # Common paths for libraries
    paths_to_check = [
        "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
        "/usr/lib/x86_64-linux-gnu/libze_loader.so",
        "/usr/local/lib/libze_loader.so"
    ]
    found_ze = False
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"[SUCCESS] Level Zero Loader found at {path}")
            found_ze = True
            break

    if not found_ze:
        print("[WARNING] Level Zero Loader not found in standard paths. Ensure 'libze1' is installed.")

    # 2. Check OpenVINO Hardware Inventory
    print("[INFO] Querying OpenVINO Runtime...")
    try:
        core = ov.Core()
        devices = core.available_devices
        print(f"[INFO] OpenVINO detected devices: {devices}")

        if "NPU" in devices:
            print("[SUCCESS] NPU is VISIBLE and READY for OpenVINO.")
            try:
                full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                print(f"      > Device Name: {full_name}")
            except:
                pass
        else:
            print("[ERROR] NPU is NOT visible to OpenVINO. Verify 'intel-level-zero-npu' driver.")

        if "GPU" in devices:
            print("[SUCCESS] Intel iGPU/dGPU is VISIBLE.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenVINO Core: {e}")

    # 3. Check for NPU Driver via clinfo (Optional)
    try:
        print("[INFO] Checking 'clinfo' for AI Boost...")
        npu_info = subprocess.check_output(["clinfo"], text=True)
        if "Intel(R) AI Boost" in npu_info or "NPU" in npu_info:
            print("[SUCCESS] clinfo confirms NPU presence.")
        else:
            print("[INFO] clinfo run successful but no explicit 'AI Boost' string found (common on some drivers).")
    except FileNotFoundError:
        print("[INFO] 'clinfo' utility not found. Skipping secondary check.")
    except Exception as e:
        print(f"[INFO] clinfo check failed: {e}")

if __name__ == "__main__":
    check_npu_visibility()
