import os
from huggingface_hub import snapshot_download

def download_full_matrix():
    # Updated to your specific project directory
    base_path = "/home/phil2/Super-Intel-NPU/current/Super-Intel-NPU/model_staging/"
    
    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # The models selected for your NPU Matrix
    models = [
        "mlabonne/Qwen3-8B-abliterated",
        "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
    ]
    
    print(f"🚀 Starting Full Repo Download to {base_path}...")

    for repo_id in models:
        # Create a clean folder name for the repo
        folder_name = repo_id.replace("/", "_")
        local_dir = os.path.join(base_path, folder_name)
        
        print(f"\n--- Fetching Full Repository: {repo_id} ---")
        print(f"Target Path: {local_dir}")
        
        try:
            # allow_patterns=None and ignore_patterns=None pulls everything
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=None,
                ignore_patterns=None 
            )
            print(f"✅ Success: {repo_id} is fully downloaded.")
        except Exception as e:
            print(f"❌ Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    # Requirement: pip install huggingface_hub
    download_full_matrix()