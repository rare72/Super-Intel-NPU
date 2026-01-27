import os
import shutil
import openvino_genai as ov_genai

# 1. Setup Paths
gguf_path = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B_INT8/dolphin-24b-q8_0.gguf"
hf_source = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B"
output_dir = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/dolphin-24B_final"

os.makedirs(output_dir, exist_ok=True)

# 2. Sync Metadata (Crucial for GGUF loaders to understand vocabulary)
print("--- Syncing Tokenizer Metadata ---")
gguf_dir = os.path.dirname(gguf_path)
for f in ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"]:
    src = os.path.join(hf_source, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(output_dir, f))
        shutil.copy(src, os.path.join(gguf_dir, f))

# 3. Perform the Bake
try:
    print(f"--- Loading GGUF: {os.path.basename(gguf_path)} ---")
    
    # We load it and tell the pipeline to save the resulting IR files
    # Note: We use 'CPU' to perform the conversion
    pipe = ov_genai.LLMPipeline(gguf_path, "CPU")
    
    print("--- Triggering Bake to OpenVINO IR ---")
    # This will generate the .xml and .bin in the current working directory or output_dir
    pipe.get_generation_config().max_new_tokens = 1
    
    # Passing the output_dir to the save method if export is missing
    # In some versions it's called save_model
    if hasattr(pipe, 'save_model'):
        pipe.save_model(output_dir)
    else:
        # Fallback: Just move any generated xml/bin to the final folder
        print("--- Manual Save Fallback ---")
        # Running a single generation often creates the files in the GGUF folder
        pipe.generate("Hi")
        for f in os.listdir(gguf_dir):
            if f.endswith((".xml", ".bin")):
                shutil.move(os.path.join(gguf_dir, f), os.path.join(output_dir, f))
    
    print(f"✅ Success! Dynamic OpenVINO model saved to: {output_dir}")
except Exception as e:
    print(f"❌ Bake Error: {e}")