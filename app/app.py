from flask import Flask, request, render_template, jsonify, send_file
import torch
from PIL import Image
import io
import os
from app.data.dataset import CustomImageDataset
from torchvision import transforms
from app.main import Config, ModelBuilder
import random
from werkzeug.utils import secure_filename
from app.predict import Predictor, CLASS_NAMES
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from datetime import datetime
import shutil
import platform

# Move this line to the top, before it's used
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='../static')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models dictionary with absolute paths
models = {
    "ResNet50": {
        "path": os.path.join(BASE_DIR, "models", "resnet50_model.pth"),
        "type": "resnet50",
    },
    "EfficientNet": {
        "path": os.path.join(BASE_DIR, "models", "efficientnet_model.pth"),
        "type": "efficientnet",
    },
}


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # Modify classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ModelManager:
    """Handles model loading and prediction."""

    def __init__(self):
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_name):
        """Load a specific model."""
        if model_name not in models:
            raise ValueError(f"Invalid model name: {model_name}")

        model_config = models[model_name]
        model_path = model_config["path"]
        model_type = model_config["type"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create model based on type
        if model_type == "efficientnet":
            # Create EfficientNet model with wrapper
            self.current_model = PlantDiseaseModel(num_classes=len(CLASS_NAMES))
        else:
            # Load ResNet50 with the correct number of classes
            self.current_model = torch.hub.load('pytorch/vision:v0.10.0', 
                                              'resnet50', 
                                              pretrained=False)
            num_features = self.current_model.fc.in_features
            self.current_model.fc = nn.Linear(num_features, len(CLASS_NAMES))

        # Load the state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different state dict formats
        if model_type == "efficientnet":
            # If the state dict has 'model' prefix, it matches our PlantDiseaseModel
            if all(k.startswith('model.') for k in state_dict.keys()):
                self.current_model.load_state_dict(state_dict)
            else:
                # If it doesn't have 'model' prefix, wrap it
                new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
                self.current_model.load_state_dict(new_state_dict)
        else:
            # For ResNet, load directly
            self.current_model.load_state_dict(state_dict)

        self.current_model = self.current_model.to(self.device)
        self.current_model.eval()
        return True

    def predict(self, image_bytes):
        """Make prediction on an image."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")

        # Prepare image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.current_model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Get the class name from CLASS_NAMES dictionary
        predicted_idx = predicted.item()
        class_name = CLASS_NAMES.get(predicted_idx, ("Unknown", "Unknown"))[1]  # Get the second element (condition)

        return {
            "label": class_name,
            "confidence": float(confidence.item()) * 100,
        }


# Initialize model manager
model_manager = ModelManager()


def get_random_image_path():
    """Get a random image path from the dataset."""
    dataset_path = os.path.join(BASE_DIR, "data", "plantDataset", "train")

    # Get all subdirectories (disease categories)
    categories = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not categories:
        return None

    # Select random category
    category = random.choice(categories)
    category_path = os.path.join(dataset_path, category)

    # Get all images in the category
    images = [
        f
        for f in os.listdir(category_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        return None

    # Select random image
    image = random.choice(images)
    return os.path.join(category_path, image)


@app.route("/")
@app.route("/index")
def home():
    models_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check which models are actually available
    available_models = {}
    for name, info in models.items():
        if os.path.exists(info["path"]):
            available_models[name] = info
            print(f"Found model: {name} at {info['path']}")  # Debug print
        else:
            print(f"Missing model: {name} at {info['path']}")  # Debug print

    template_data = {
        'model_names': list(available_models.keys()),
        'total_models': len(available_models),
        'model_dir': models_dir,
        'error': None if available_models else "No trained models found. Please train models first.",
        'class_names': CLASS_NAMES
    }

    # Get initial random background
    image_path = get_random_image_path()
    if image_path:
        background_url = f'/serve_image/{os.path.relpath(image_path, BASE_DIR).replace(os.sep, "_")}'
        template_data['background_url'] = background_url

    return render_template('index.html', **template_data)


@app.route("/load_model", methods=["POST"])
def load_model():
    try:
        model_name = request.form.get("model_name")
        success = model_manager.load_model(model_name)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form.get('model')
        if not model_name:
            return render_template('prediction_result.html', error='No model selected')
            
        model_info = models.get(model_name)
        if not model_info:
            return render_template('prediction_result.html', error='Invalid model selected')
            
        # Load the model using ModelManager
        model_manager.load_model(model_name)

        if 'file' not in request.files:
            return render_template('prediction_result.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('prediction_result.html', error='No file selected')

        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the upload folder exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        file.save(file_path)

        # Read the file for prediction
        with open(file_path, 'rb') as f:
            file_contents = f.read()
        
        # Use ModelManager's predict method
        prediction = model_manager.predict(file_contents)
        
        return render_template('prediction_result.html', 
                             class_name=prediction['label'],
                             confidence=prediction['confidence'],
                             image_name=filename,
                             model_used=model_name)

    except Exception as e:
        return render_template('prediction_result.html', error=str(e))


@app.route("/predict_batch", methods=['POST'])
def predict_batch():
    try:
        predictor = Predictor(model_name="best_model.pth")

        # Get multiple images from request
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        predictions = []
        for file in files:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(temp_path)
            
            class_id, confidence = predictor.predict(temp_path)
            predictions.append({
                'filename': file.filename,
                'class_id': class_id,
                'confidence': confidence * 100
            })
            
            os.remove(temp_path)

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/get_random_background")
def get_random_background():
    """API endpoint to get a random background image."""
    image_path = get_random_image_path()
    if not image_path:
        return jsonify({"error": "No images found"}), 404

    # Convert the file path to a URL
    relative_path = os.path.relpath(image_path, BASE_DIR)
    background_url = f'/serve_image/{relative_path.replace(os.sep, "_")}'
    return jsonify({"background_url": background_url})


@app.route("/serve_image/<path:filename>")
def serve_image(filename):
    """Serve the image file."""
    # Convert URL-safe path back to system path
    real_path = os.path.join(BASE_DIR, *filename.split("_"))
    return send_file(real_path)


@app.route("/wh")
def wh_page():
    return render_template('wh.html')


@app.route("/contribute", methods=["POST"])
def contribute():
    try:
        disease_label = request.form.get('disease_label')
        if not disease_label:
            return render_template('index.html', error='Please select a disease label')

        if 'training_images' not in request.files:
            return render_template('index.html', error='No files uploaded')

        files = request.files.getlist('training_images')
        if not files or files[0].filename == '':
            return render_template('index.html', error='No files selected')

        # Define the dataset path
        dataset_path = os.path.join(BASE_DIR, "data", "plantDataset", "train", disease_label)
        os.makedirs(dataset_path, exist_ok=True)

        saved_files = []
        for file in files:
            if file and file.filename:
                # Generate a unique filename using timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                original_extension = os.path.splitext(file.filename)[1]
                new_filename = f"{disease_label}_{timestamp}{original_extension}"
                
                # Save the file to the dataset directory
                file_path = os.path.join(dataset_path, new_filename)
                file.save(file_path)
                saved_files.append(new_filename)

        success_message = f"Successfully contributed {len(saved_files)} images to the {disease_label} dataset"
        return render_template('index.html', success=success_message)

    except Exception as e:
        return render_template('index.html', error=f'Error contributing images: {str(e)}')


@app.route("/debug")
def debug():
    """Debug page showing detailed system information."""
    try:
        models_dir = os.path.join(BASE_DIR, "models")
        plots_dir = os.path.join(BASE_DIR, "static", "plots")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Get all training plot files
        plot_files = {}
        if os.path.exists(plots_dir):
            for model_name in ["ResNet50", "EfficientNet"]:
                model_plots = [f for f in os.listdir(plots_dir) 
                             if f.startswith(f'training_loss_{model_name.lower()}')]
                if model_plots:
                    # Get the most recent plot
                    latest_plot = sorted(model_plots)[-1]
                    plot_files[model_name] = url_for('static', 
                                                    filename=f'plots/{latest_plot}')
        
        # Check model files and their status
        model_status = {}
        for name, info in models.items():
            exists = os.path.exists(info["path"])
            model_status[name] = {
                "path": info["path"],
                "type": info["type"],
                "exists": exists,
                "plot": plot_files.get(name, None)
            }
        
        template_data = {
            'model_names': list(models.keys()),
            'total_models': len([m for m in model_status.values() if m["exists"]]),
            'model_dir': models_dir,
            'models': model_status,
            'error': None if any(m["exists"] for m in model_status.values()) 
                    else "No trained models found in models directory",
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        }
        
        return render_template('debug.html', **template_data)
        
    except Exception as e:
        print(f"Error in debug route: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template(
            'debug.html',
            error=str(e),
            model_names=[],
            models={},
            model_dir=models_dir,
            total_models=0
        )


if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    
    # Check if model files exist
    for model_name, model_info in models.items():
        if not os.path.exists(model_info["path"]):
            print(f"Warning: Model file not found: {model_info['path']}")
    
    app.run(debug=True)
