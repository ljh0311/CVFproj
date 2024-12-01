import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import os
from flask import request, jsonify
from werkzeug.utils import secure_filename
import glob
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from app.models.model import ModelBuilder  # Import ModelBuilder
from app.constants import MODEL_DIR, UPLOAD_FOLDER
from app.config import Config  # Add this import

# Define constants
MODEL_DIR = os.path.join(Config.BASE_DIR, "models")  # Update this to use Config
UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, "temp", "uploads")  # Update this too

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add default CLASS_NAMES at the top of the file
DEFAULT_CLASS_NAMES = {
    0: ("Apple", "Apple Scab"),
    1: ("Apple", "Black Rot"),
    2: ("Apple", "Cedar Apple Rust"), 
    3: ("Apple", "Healthy"),
    4: ("Blueberry", "Healthy"),
    5: ("Cherry (including sour)", "Healthy"),
    6: ("Cherry (including sour)", "Powdery Mildew"),
    7: ("Corn (maize)", "Cercospora Leaf Spot Gray Leaf Spot"),
    8: ("Corn (maize)", "Common Rust"),
    9: ("Corn (maize)", "Healthy"),
    10: ("Corn (maize)", "Northern Leaf Blight"),
    11: ("Grape", "Black Rot"),
    12: ("Grape", "Esca (Black Measles)"),
    13: ("Grape", "Healthy"),
    14: ("Grape", "Leaf Blight (Isariopsis Leaf Spot)"),
    15: ("Orange", "Haunglongbing (Citrus Greening)"),
    16: ("Peach", "Bacterial Spot"),
    17: ("Peach", "Healthy"),
    18: ("Pepper Bell", "Bacterial Spot"),
    19: ("Pepper Bell", "Healthy"),
    20: ("Potato", "Early Blight"),
    21: ("Potato", "Healthy"),
    22: ("Potato", "Late Blight"),
    23: ("Raspberry", "Healthy"),
    24: ("Soybean", "Healthy"),
    25: ("Squash", "Powdery Mildew"),
    26: ("Strawberry", "Healthy"),
    27: ("Strawberry", "Leaf Scorch"),
    28: ("Tomato", "Bacterial Spot"),
    29: ("Tomato", "Early Blight"),
    30: ("Tomato", "Healthy"),
    31: ("Tomato", "Late Blight"),
    32: ("Tomato", "Leaf Mold"),
    33: ("Tomato", "Septoria Leaf Spot"),
    34: ("Tomato", "Spider Mites Two-spotted Spider Mite"),
    35: ("Tomato", "Target Spot"),
    36: ("Tomato", "Tomato Mosaic Virus"),
    37: ("Tomato", "Tomato Yellow Leaf Curl Virus"),
}

def load_class_names():
    """Load class names from file."""
    class_names_path = os.path.join(Config.BASE_DIR, "models", "class_names.txt")
    class_names = {}
    
    try:
        with open(class_names_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:  # Only process lines that contain a colon
                    idx, name = line.split(':', 1)  # Split on first colon only
                    class_names[int(idx)] = name
                else:
                    print(f"Warning: Skipping malformed line in class_names.txt: {line}")
    except FileNotFoundError:
        print(f"Warning: Class names file not found at {class_names_path}")
        return {}
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return {}
        
    return class_names

# Now define CLASS_NAMES
CLASS_NAMES = load_class_names()


def get_available_models():
    """Get list of available model files."""
    model_dir = os.path.join(Config.BASE_DIR, "models")
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return []
        
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    print(f"Found model files: {model_files}")  # Debug print
    return [os.path.basename(f) for f in model_files]


class Predictor:
    def __init__(self, model_path, model_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_type = model_type
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
        
    def _load_model(self):
        if self.model_type == "resnet50":
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        elif self.model_type == "efficientnet":
            model = efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_ftrs, len(CLASS_NAMES))
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return predicted.item(), confidence.item()

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise


def main():
    """Main prediction function."""
    try:
        available_models = get_available_models()
        if not available_models:
            return jsonify({"error": "No models available"}), 500

        selected_model = request.form.get("model")
        if not selected_model:
            selected_model = available_models[-1]
        elif selected_model not in available_models:
            return jsonify({"error": "Invalid model selected"}), 400

        # Determine model type
        model_type = "efficientnet" if "efficientnet" in selected_model.lower() else "resnet18"
        print(f"Selected model: {selected_model} (Type: {model_type})")

        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        allowed_extensions = {"png", "jpg", "jpeg"}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return (
                jsonify(
                    {"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed"}
                ),
                400,
            )

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        predictor = Predictor(model_path=os.path.join(MODEL_DIR, selected_model), model_type=model_type)
        predicted_class, confidence = predictor.predict(temp_path)

        os.remove(temp_path)

        return jsonify({
            "class": int(predicted_class),
            "class_name": CLASS_NAMES.get(predicted_class, "Unknown"),
            "confidence": float(confidence),
            "model_used": selected_model,
            "model_type": model_type,
            "available_models": available_models,
        })

    except Exception as e:
        print(f"Error in main: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_models():
    """Return list of available models."""
    try:
        models = get_available_models()
        return jsonify(
            {"models": models, "default_model": models[-1] if models else None}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    main()
