import os
import torch
import urllib.request

def download_model():
    # Create checkpoints directory if it doesn't exist
    os.makedirs(r"C:\Users\HP\checkpoints", exist_ok=True)
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = r"C:\Users\HP\checkpoints\sam_vit_b_01ec64.pth"
    
    if not os.path.exists(model_path):
        print("Downloading model... (This may take a few minutes)")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded to: {model_path}")
    else:
        print("Model already exists at:", model_path)

if __name__ == "__main__":
    download_model()