import os
import torch
from pathlib import Path
import random
from PIL import Image
import torch
from torchvision import transforms
from app.config import Config
from app.models.model import ModelBuilder
from app.predict import DEFAULT_CLASS_NAMES

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
                print(f"[OK] {model_type} ({filename}): {size_mb:.2f} MB - Successfully loaded")
            except Exception as e:
                print(f"[X] {model_type} ({filename}): {size_mb:.2f} MB - Error loading: {str(e)}")
        else:
            print(f"[X] {model_type} ({filename}): Not found")

def test_predictions():
    """Test model predictions on random images from both datasets."""
    print("\nTesting model predictions...")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    # Load the latest model
    model_dir = os.path.join(Config.BASE_DIR, "models")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print("No trained models found!")
        return
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    try:
        # Load model
        model = ModelBuilder.create_model(len(DEFAULT_CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()
        model.to(Config.DEVICE)
        
        # Test plant dataset
        plant_dir = os.path.join(Config.BASE_DIR, "data", "plantDataset", "test")
        if os.path.exists(plant_dir):
            # Get random class directory
            class_dirs = [d for d in os.listdir(plant_dir) if os.path.isdir(os.path.join(plant_dir, d))]
            if class_dirs:
                random_class = random.choice(class_dirs)
                class_path = os.path.join(plant_dir, random_class)
                
                # Get random image
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    
                    # Make prediction
                    print(f"\nTesting plant image: {random_image}")
                    print(f"True class: {random_class}")
                    
                    image = Image.open(image_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                    predicted_class = DEFAULT_CLASS_NAMES[predicted.item()]
                    print(f"Predicted: {predicted_class}")
                    print(f"Confidence: {confidence.item():.2%}")
        
        # Test landscape dataset
        landscape_dir = os.path.join(Config.BASE_DIR, "data", "landscapeDataset")
        if os.path.exists(landscape_dir):
            # Get random image from landscape dataset
            images = []
            for root, _, files in os.walk(landscape_dir):
                images.extend([os.path.join(root, f) for f in files 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if images:
                random_image = random.choice(images)
                print(f"\nTesting landscape image: {os.path.basename(random_image)}")
                print("True class: Landscape")
                
                image = Image.open(random_image).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                predicted_class = DEFAULT_CLASS_NAMES[predicted.item()]
                print(f"Predicted: {predicted_class}")
                print(f"Confidence: {confidence.item():.2%}")
        
    except Exception as e:
        print(f"Error during prediction testing: {str(e)}")

if __name__ == "__main__":
    verify_setup()
    test_predictions() 