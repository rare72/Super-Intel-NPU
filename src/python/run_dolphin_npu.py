import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# 1. Path to your INT8 Baked Dolphin 24B
model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B_final"

# 2. Optimized NPU 3 Configuration
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_COMPILER_TYPE": "DRIVER",
    "NPUW_LLM_PREFILL_CHUNK_SIZE": "512" 
}

print(f"--- Initializing Dolphin 24B (Ironclad Static Mode) ---")

# 3. Load Metadata & Weights
tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)

# compile=False is required to apply the static override
model = OVModelForCausalLM.from_pretrained(
    model_dir, 
    device="NPU", 
    ov_config=ov_config,
    trust_remote_code=True,
    compile=False 
)

# 4. THE FIX: Explicit Static Shape Mapping
# We are killing ALL dynamic dimensions. The NPU will allocate exactly 
# enough space in your 64GB RAM for a 1024-token context.
print("--- Overriding Memory to Static 1024 Token Window ---")
# 1024 is safer for the first run of a 24B model to ensure stability
model.model.reshape({
    "input_ids": [1, 1024],
    "attention_mask": [1, 1024],
    "position_ids": [1, 1024]
})

# 5. NPU Compilation
print("--- Compiling for NPU 3 (Mapping 24B Parameters...) ---")
model.compile()

# 6. Streamer
streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\n--- Setup Complete. Dolphin 24B is live on NPU. ---")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Dolphin / ChatML Prompt Format
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    # We must pad the input to exactly 1024 tokens to match the static shape
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=1024, 
        truncation=True
    )

    print("\nAssistant: ", end="")
    model.generate(
        **inputs, 
        max_new_tokens=256, 
        streamer=streamer,
        do_sample=True,
        temperature=0.7
    )
    print("\n")