import os
import time
import glob
from huggingface_hub import HfApi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

REPO_ID = "auto-cap/moe-cap-results"

POLL_INTERVAL = 10


def get_api_client():
    token = os.environ.get("HF_TOKEN")
    if token:
        print("[Watcher] Detected HF_TOKEN environment variable, authenticating...")
        return HfApi(token=token)
    else:
        print("[Watcher] HF_TOKEN environment variable not found.")
        print("   Attempting to initialize without token (may fail for uploads)...")
        return HfApi()

def main():
    api = get_api_client()
    uploaded_files = set()

    print(f"Starting directory monitor: {LOCAL_OUTPUT_DIR}")
    print(f"Target repository: {REPO_ID}")


    if not os.path.exists(LOCAL_OUTPUT_DIR):
        print(f"Directory {LOCAL_OUTPUT_DIR} does not exist yet. Waiting...")
        while not os.path.exists(LOCAL_OUTPUT_DIR):
            time.sleep(5)
        print(f"Directory found: {LOCAL_OUTPUT_DIR}")

    try:
        print("Initializing file list...", end="")
        initial_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "**/*.json"), recursive=True)
        for f in initial_files:
            uploaded_files.add(os.path.abspath(f))
        print(f" Found {len(uploaded_files)} existing files (skipping upload).")
    except Exception as e:
        print(f" (Scan warning: {e})")

    while True:
        try:
            current_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "**/*.json"), recursive=True)
            
            for file_path in current_files:
                abs_path = os.path.abspath(file_path)
                
                if abs_path not in uploaded_files:
                    relative_path = os.path.relpath(abs_path, LOCAL_OUTPUT_DIR)
                    
                    print(f"New file found: {relative_path}")
                    print(f"   Uploading...", end="", flush=True)
                    
                    try:
                        api.upload_file(
                            path_or_fileobj=abs_path,
                            path_in_repo=relative_path, 
                            repo_id=REPO_ID,
                            repo_type="dataset"
                        )
                        uploaded_files.add(abs_path)
                        print(" Success")
                    except Exception as upload_err:
                        print(f" Failed: {upload_err}")
        
        except Exception as e:
            print(f"Polling error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()