import os
import argparse
from huggingface_hub import snapshot_download

def download_qwen(model_id="mlabonne/Qwen3-8B-abliterated", staging_dir="model/model_template"):
    print(f"--- Qwen3 Downloader ---")
    print(f"Target Model: {model_id}")
    print(f"Staging Dir:  {staging_dir}")

    # Model Cache Path
    model_name_clean = model_id.split("/")[-1]
    hf_cache_path = f"/Super-Intel-NPU/cache/model_{model_name_clean}"
    print(f"Cache Dir:    {hf_cache_path}")

    if os.path.exists(staging_dir):
        print(f"[Info] Staging directory exists. Resuming download...")
    else:
        os.makedirs(staging_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=staging_dir,
            cache_dir=hf_cache_path,
            local_dir_use_symlinks=False, # Important for NNCF
            resume_download=True
        )
        print(f"[Success] Model downloaded to {staging_dir}")
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        exit(1)

if __name__ == "__main__":
    download_qwen()
