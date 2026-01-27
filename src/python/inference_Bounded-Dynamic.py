import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Path where you saved the model in Step 1
model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging_qwen_exp1"
# 1. Config for Arrow Lake NPU 3
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_COMPILER_TYPE": "DRIVER"
}

print("--- Loading Baked INT8 Model ---")
# We load from disk with compile=False to perform the 'Manual Reshape'
model = OVModelForCausalLM.from_pretrained(
    model_dir, 
    device="NPU", 
    compile=False, 
    ov_config=ov_config
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 2. THE FIX: Explicitly Bound the Input Tensors
# This replaces the 'infinite' shape with a 2048-token ceiling.
# This satisfies the NPU driver and protects your 12.8 GiB limit.
print("--- Applying 2048 Token Memory Bound ---")
new_shapes = {
    "input_ids": ov.PartialShape([1, ov.Dimension(1, 2048)]),
    "attention_mask": ov.PartialShape([1, ov.Dimension(1, 2048)])
}
model.model.reshape(new_shapes)

# 3. Compile for NPU 3
print("--- Compiling for Arrow Lake NPU ---")
model.compile()

# 4. Chat Loop
print("\n--- Setup Successful. Type your prompt ---")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]: break

    inputs = tokenizer(user_input, return_tensors="pt")
    print("\nAssistant: ", end="")
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))