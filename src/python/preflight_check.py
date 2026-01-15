import os
import openvino as ov
import sys
import argparse

def run_audit(model_path):
    print("--- New Offering: NPU Pre-Flight Audit ---")

    # Resolve Path
    if os.path.isdir(model_path):
        # If directory, look for openvino_model.xml
        xml_check = os.path.join(model_path, "openvino_model.xml")
        if os.path.exists(xml_check):
            model_path = xml_check
        else:
             print(f"[Audit] Error: openvino_model.xml not found in directory: {model_path}")
             return False

    print(f"[Audit] Checking Model: {model_path}")

    # 1. Size Check (The 'Smoking Gun' Fix)
    bin_path = model_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size_bytes = os.path.getsize(bin_path)
        size_gb = size_bytes / (1024**3)
        print(f"[Audit] Binary Size: {size_gb:.2f} GB")

        # 4GB is roughly the limit for INT4 7B, 14GB is FP16
        if size_gb > 6.0:
            print(f"!! CRITICAL ERROR: Model size ({size_gb:.2f}GB) indicates FP16 weights.")
            print("!! ACTION: Re-run the NNCF Bake command. Do NOT load this onto the NPU.")
            return False
        print("[SUCCESS] Model size is within INT4 safety limits (~4-5GB).")
    else:
        print(f"[Audit] Warning: .bin file not found at {bin_path}")
        # Proceeding might fail, but let OpenVINO core decide

    # 2. Metadata & Precision Check
    try:
        core = ov.Core()
        # Verify file existence explicitly before OpenVINO tries (better error msg)
        if not os.path.exists(model_path):
             print(f"[Audit] Critical: XML file does not exist: {model_path}")
             return False

        model = core.read_model(model_path)
        print(f"[Audit] Model loaded successfully for inspection.")

        # 3. Static Shape Check (The 'Dynamic' Error Fix)
        is_static = True
        for input_layer in model.inputs:
            shape = input_layer.get_partial_shape()
            print(f"[Audit] Input '{input_layer.any_name}' Shape: {shape}")
            if shape.is_dynamic:
                print(f"!! WARNING: Dynamic shape detected in '{input_layer.any_name}'.")
                is_static = False

        if not is_static:
            print("!! ACTION: Model is DYNAMIC. You must call .reshape([1, 128]) before compiling.")
            # We allow it to proceed if the supervisor handles reshaping, but warn heavily.
        else:
            print("[SUCCESS] Model is STATIC and NPU-ready.")

    except Exception as e:
        print(f"!! FATAL: OpenVINO could not read model: {e}")
        return False

    print("--- Audit Complete: System Green for NPU Load ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New Offering Pre-Flight Audit")
    parser.add_argument("model_path", type=str, nargs='?', default="./models/neuralchat_int4/openvino_model.xml",
                        help="Path to the OpenVINO XML model file or directory")
    args = parser.parse_args()

    if not run_audit(args.model_path):
        sys.exit(1)
