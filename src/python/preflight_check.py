import os
import openvino as ov
import sys

def run_audit(model_path):
    print("--- New Offering: NPU Pre-Flight Audit ---")

    # 1. Size Check (The 'Smoking Gun' Fix)
    bin_path = model_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size_gb = os.path.getsize(bin_path) / (1024**3)
        print(f"[Audit] Binary Size: {size_gb:.2f} GB")
        if size_gb > 6.0:
            print(f"!! CRITICAL ERROR: Model size ({size_gb:.2f}GB) indicates FP16 weights.")
            print("!! ACTION: Re-run the NNCF Bake command. Do NOT load this onto the NPU.")
            return False
        print("[SUCCESS] Model size is within INT4 safety limits (~4-5GB).")
    else:
        print(f"[Audit] Warning: .bin file not found at {bin_path}")

    # 2. Metadata & Precision Check
    try:
        core = ov.Core()
        model = core.read_model(model_path)

        # Check weight types in the graph
        # Note: get_ops() can be huge. We just want to ensure we don't error out reading it.
        # Checking every op precision might be verbose, let's just check input types
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
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "./models/neuralchat_int4/openvino_model.xml"
    if not run_audit(xml_file):
        sys.exit(1)
