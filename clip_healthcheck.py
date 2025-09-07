import os
import sys
from transformers import CLIPModel, CLIPProcessor

def main():
    model_path = os.getenv("CLIP_MODEL_PATH")
    if not model_path:
        print("[healthcheck] ❌ CLIP_MODEL_PATH not set")
        sys.exit(1)

    try:
        print(f"[healthcheck] Checking CLIP model at {model_path}...")
        CLIPModel.from_pretrained(model_path, local_files_only=True)
        CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        print("[healthcheck] ✅ CLIP model loaded successfully (offline mode).")
    except Exception as e:
        print(f"[healthcheck] ❌ Failed to load CLIP model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
