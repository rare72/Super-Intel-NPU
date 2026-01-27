import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Path where you saved the model in Step 1
#model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging_qwen_exp1"
model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/models/qwen3_baked_int8_exp3"

ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_COMPILER_TYPE": "DRIVER"
}

print("--- Step 2: Loading Baked Model from Disk ---")
# compile=False is required so we can apply the memory bound
model = OVModelForCausalLM.from_pretrained(
    model_dir, 
    device="NPU", 
    compile=False, 
    ov_config=ov_config
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# THE CRITICAL FIX: Direct Graph Reshape
# This manually replaces the 'infinite' bounds with a 2048-token limit.
# This satisfies the NPU 3 driver and protects your 12.8 GiB RAM limit.
print("--- Applying NPU Memory Boundary (2048 Tokens) ---")
model.model.reshape({
    "input_ids": ov.PartialShape([1, ov.Dimension(1, 2048)]),
    "attention_mask": ov.PartialShape([1, ov.Dimension(1, 2048)])
})

print("--- Compiling for NPU 3 (This takes ~60 seconds) ---")
model.compile()

# Chat Loop
print("\n--- Setup Successful. ---")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]: break

    inputs = tokenizer(user_input, return_tensors="pt")
    print("\nAssistant: ", end="")
    # max_new_tokens + prompt length must be < 2048
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))