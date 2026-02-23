import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip
import requests
from pathlib import Path

# Use environment variables for cache directories if set, else use local defaults
HF_HOME = os.environ.get('HF_HOME', 'app/models/huggingface')
TORCH_HOME = os.environ.get('TORCH_HOME', 'app/models/torch')

def download_blip():
    print("Downloading BLIP model...")
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    # Save to local directory
    save_path = "app/models/blip"
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"BLIP model saved to {save_path}")

def download_openclip():
    print("Downloading OpenCLIP model...")
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    
    # This downloads the weights to TORCH_HOME/hub/checkpoints or similar
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    print(f"OpenCLIP {model_name} downloaded.")

def download_phi3():
    print("Downloading Phi-3 Mini GGUF...")
    url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    save_path = Path("app/models/llm/phi3-mini-instruct-q4.gguf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not save_path.exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Phi-3 GGUF saved to {save_path}")
    else:
        print("Phi-3 GGUF already exists.")

if __name__ == "__main__":
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(TORCH_HOME, exist_ok=True)
    download_blip()
    download_openclip()
    download_phi3()
