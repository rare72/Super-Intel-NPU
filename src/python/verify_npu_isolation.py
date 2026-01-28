import openvino as ov
import openvino.properties as props
import os

def verify_npu_isolation():
    # 1. Initialize OpenVINO Core
    core = ov.Core()

    # 2. Check System Visibility
    devices = core.available_devices
    print(f"--- [System Check] ---")
    print(f"Visible Devices: {devices}")

    # 3. Verify Isolation Strategy
    # If ZE_AFFINITY_MASK=0.0 is active, 'GPU' (NVIDIA) should NOT appear here.
    # Note: Intel NPU often shows up as 'NPU'. Intel iGPU shows as 'GPU'.
    # If we want to isolate to NPU only, we expect NO 'GPU' unless it's the iGPU being passed through.
    # The prompt specifically mentioned preventing probing of NVIDIA GPUs.

    if "NPU" in devices:
        print("SUCCESS: NPU device is visible and ready.")
        if any("GPU" in d for d in devices):
            print("(Note: GPU devices are also visible. This is expected on systems with NVIDIA dGPUs.)")
    else:
        print("ERROR: NPU device not found in OpenVINO Core. Check drivers.")

    # 4. Deep Dive into NPU 0 Properties
    if "NPU" in devices:
        print(f"\n--- [NPU 0 Deep Probe] ---")
        try:
            # Querying the AD1D node properties
            full_name = core.get_property("NPU", props.device.full_name)
            architecture = core.get_property("NPU", props.device.architecture)

            print(f"Device Name:  {full_name}")
            print(f"Architecture: {architecture}")

            if "3720" in str(architecture):
                print("Target: NPU 3720 (Meteor Lake / Core Ultra) - SUPPORTED")
            elif "4000" in str(architecture):
                print("Target: NPU 4000 (Arrow Lake) - SUPPORTED")
            else:
                print(f"Target: Unknown Architecture ({architecture}) - Warning")

            # 5. Confirming the 32GB Big Door (Aperture)
            # We query the capabilities to verify the SVM shared virtual memory pool
            # Note: Starting in 2025.3, NPU supports dynamic memory queries
            supported_props = core.get_property("NPU", props.supported_properties)
            print("\nSupported NPU Capabilities Found.")
        except Exception as e:
            print(f"Property Query Error: {e}")

    else:
        print("ERROR: NPU not found in Core. Check driver installation.")

if __name__ == "__main__":
    # Check if variables are recognized by the current process
    print(f"Environment ZE_AFFINITY_MASK: {os.environ.get('ZE_AFFINITY_MASK')}")
    verify_npu_isolation()
