import os
import argparse
from huggingface_hub import snapshot_download

def download_qwen(model_id="mlabonne/Qwen3-8B-abliterated", staging_dir="./model_staging_qwen"):
    print(f"--- Qwen3 Downloader ---")
    print(f"Target Model: {model_id}")
    print(f"Staging Dir:  {staging_dir}")

    if os.path.exists(staging_dir):
        print(f"[Info] Staging directory exists. Resuming download...")
    else:
        os.makedirs(staging_dir)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=staging_dir,
            local_dir_use_symlinks=False, # Important for NNCF
            resume_download=True
        )
        print(f"[Success] Model downloaded to {staging_dir}")
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        exit(1)

if __name__ == "__main__":
    download_qwen()
