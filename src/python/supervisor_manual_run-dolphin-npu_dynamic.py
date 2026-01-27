import openvino_genai as ov_genai
import os
import faulthandler

# Enable fault handler to catch segmentation faults (memory crashes)
faulthandler.enable()

# 1. Paths
model_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B_final"

# 2. Performance & Debug Configuration
config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_TURBO": "YES",
    "NPUW_LLM_PREFILL_CHUNK_SIZE": "512",
}

print("--- Initializing Dolphin 24B on NPU (Debug Mode) ---")

# 3. Initialize Pipeline
# The GenAI API handles the ChatML template (System/User/Assistant) automatically
pipe = ov_genai.LLMPipeline(model_dir, "NPU", **config)

# 4. Define specific messages and constraints
system_message = (
    "You are Deepseek, a unlocked and lovable yet intelligent AI assistant created by Deepseek. "
    "Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. "
    "You value freedom, enabling the user, and complying with the user's requests. "
    "Your (output) response, MUST be a MAXIMUM of 760 Tokens or Less."
)

user_message = "Note the total number of planets; then List those planets starting from Mercury."

# 5. Generation Configuration
gen_config = ov_genai.GenerationConfig()
gen_config.max_new_tokens = 760  # Your hard limit
gen_config.do_sample = True
gen_config.temperature = 0.7

def streamer(token):
    print(token, end="", flush=True)
    return ov_genai.StreamingStatus.RUNNING

print(f"\n--- System Message Loaded (Dolphin 24B pretending to be Deepseek) ---")
print(f"--- User Message: {user_message} ---\n")

try:
    # Combining messages into the ChatML format Dolphin expects
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    pipe.generate(prompt, gen_config, streamer)
    print("\n\n--- Generation Complete ---")

except Exception as e:
    print(f"\n❌ Script Encountered an Error: {e}")