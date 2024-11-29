import os
import torch
from pathlib import Path

def verify_setup():
    # Paths
    base_dir = r"C:\Users\user\Documents\SITstuffs\CompVis\CVFproj"
    model_dir = os.path.join(base_dir, "models")
    
    print("\n=== Directory Structure ===")
    print(f"Base Directory: {base_dir}")
    print(f"Models Directory: {model_dir}")
    
    # Check directories
    print("\n=== Directory Checks ===")
    print(f"Base directory exists: {os.path.exists(base_dir)}")
    print(f"Models directory exists: {os.path.exists(model_dir)}")
    
    # List files in models directory
    if os.path.exists(model_dir):
        print("\n=== Model Files ===")
        for file in os.listdir(model_dir):
            path = os.path.join(model_dir, file)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"- {file} ({size_mb:.2f} MB)")
    
    # Expected model files
    expected_models = {
        "resnet50_model.pth": "ResNet50",
        "plant_disease_model.pth": "EfficientNet"
    }
    
    print("\n=== Model File Checks ===")
    for filename, model_type in expected_models.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            try:
                # Try to load the model file
                state_dict = torch.load(path, map_location='cpu')
                print(f"✓ {model_type} ({filename}): {size_mb:.2f} MB - Successfully loaded")
            except Exception as e:
                print(f"✗ {model_type} ({filename}): {size_mb:.2f} MB - Error loading: {str(e)}")
        else:
            print(f"✗ {model_type} ({filename}): Not found")

if __name__ == "__main__":
    verify_setup() 