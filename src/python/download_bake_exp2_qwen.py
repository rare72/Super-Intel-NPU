from optimum.intel import OVModelForCausalLM

# Use the model you wanted
model_id = "mlabonne/Qwen3-8B-abliterated"
save_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/models/qwen3_baked_int8_exp3"

print("--- Exporting and Compressing Model to INT8 (Stay below 12.8GB) ---")
model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True, 
    weight_format="int8", 
    trust_remote_code=True
)

# Save it so we never have to 'export' again
model.save_pretrained(save_dir)
print(f"--- Done! Model saved to {save_dir} ---")