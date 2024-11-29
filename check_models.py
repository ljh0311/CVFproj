import os

BASE_DIR = r"C:\Users\user\Documents\SITstuffs\CompVis\CVFproj"
MODEL_DIR = os.path.join(BASE_DIR, "models")

def check_models():
    print(f"Checking models in: {MODEL_DIR}")
    print(f"Directory exists: {os.path.exists(MODEL_DIR)}")
    
    expected_models = {
        "resnet50_model.pth": "ResNet50",
        "plant_disease_model.pth": "EfficientNet"
    }
    
    if os.path.exists(MODEL_DIR):
        print("\nFiles in models directory:")
        for file in os.listdir(MODEL_DIR):
            file_path = os.path.join(MODEL_DIR, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"- {file} ({size_mb:.2f} MB)")
            
        print("\nChecking expected models:")
        for filename, model_name in expected_models.items():
            path = os.path.join(MODEL_DIR, filename)
            exists = os.path.exists(path)
            if exists:
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"{model_name}: {path} (Exists: {exists}, Size: {size_mb:.2f} MB)")
            else:
                print(f"{model_name}: {path} (Exists: {exists})")

if __name__ == "__main__":
    check_models() 