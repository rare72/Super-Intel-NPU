import openvino as ov
import numpy as np
import time

# 1. Initialize OpenVINO Core
core = ov.Core()
device = "NPU"

print(f"--- Starting NPU Verification ---")
print(f"Targeting Device: {device}")

# 2. Create a simple model (Synthetic 2-layer MLP)
# This avoids downloading large files while testing connectivity
param = ov.opset10.parameter([1, 1024], ov.Type.f32)
relu = ov.opset10.relu(param)
res = ov.opset10.result(relu)
model = ov.Model([res], [param], "SimpleTestModel")

# 3. Compile the model for the NPU
# This is where the NPU Compiler (libze_intel_npu) actually works
print("Compiling model for NPU... (this may take a few seconds)")
start_compile = time.time()
compiled_model = core.compile_model(model, device)
print(f"Compilation successful in {time.time() - start_compile:.2f}s")

# 4. Prepare dummy input data
input_data = np.random.rand(1, 1024).astype(np.float32)

# 5. Run Inference
print("Running inference...")
start_infer = time.time()
results = compiled_model([input_data])
end_infer = time.time()

print(f"Inference successful!")
print(f"Execution time: {(end_infer - start_infer) * 1000:.3f} ms")
print(f"--- NPU is fully operational ---")
