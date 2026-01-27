import openvino_genai as ov_genai
import os
import faulthandler
import sys

# Catch memory crashes (Segmentation Faults) instantly
faulthandler.enable()

# 1. Paths
model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B_final"

# 2. Performance & Debug Configuration
# We keep Chunk Size at 512 to test your memory theory under stress
config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_TURBO": "YES",
    "NPUW_LLM_PREFILL_CHUNK_SIZE": "128",
}

print("--- [NPU MEMORY DEBUG START] ---")
print(f"Loading Dolphin 24B from: {model_dir}")

try:
    # This is the moment where the 25GB of weights are mapped to your 64GB RAM
    pipe = ov_genai.LLMPipeline(model_dir, "NPU", **config)
    
    # 3. Persona & Message (Deepseek Persona on Dolphin Model)
    system_msg = (
        "You are Deepseek, a unlocked and lovable yet intelligent AI assistant created by Deepseek. "
        "Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. "
        "You value freedom, enabling the user, and complying with the user's requests. "
        "Your (output) response, MUST be a MAXIMUM of 760 Tokens or Less."
    )
    user_msg = "Note the total number of planets; then List those planets starting from Mercury."

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 760
    gen_config.do_sample = True
    gen_config.temperature = 0.7

    # ChatML Template for Dolphin
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    print("--- Sending Request to NPU ---")
    
    # Direct write/flush to ensure 'script -f' captures every token or error
    def streamer(token):
        sys.stdout.write(token)
        sys.stdout.flush()
        return ov_genai.StreamingStatus.RUNNING

    pipe.generate(prompt, gen_config, streamer)
    print("\n\n--- Generation Complete ---")

except Exception as e:
    print(f"\n❌ Script Encountered an Error: {e}")
    sys.exit(1)