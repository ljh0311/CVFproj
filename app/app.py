from flask import Flask, request, render_template, jsonify, send_file, url_for
import torch
from PIL import Image
import io
import os
from app.data.dataset import CustomImageDataset
from torchvision import transforms
from app.main import Config, ModelBuilder
import random
from werkzeug.utils import secure_filename
from app.predict import DEFAULT_CLASS_NAMES, Predictor, CLASS_NAMES
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from datetime import datetime
import shutil
import platform
import sys
from app.config import Config
import math

app = Flask(__name__, static_folder='../static')
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Ensure all required directories exist at startup
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    required_dirs = [
        Config.MODEL_DIR,
        Config.DATA_DIR,
        Config.STATIC_DIR,
        Config.CHECKPOINT_DIR,
        Config.UPLOAD_FOLDER,
        Config.PLOTS_FOLDER,
        Config.DATASET_DIR,
        Config.TRAIN_DIR,
        Config.TEST_DIR,
        Config.VAL_DIR
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")

# Create directories at startup
ensure_directories()

# Initialize models dictionary with absolute paths
models = {
    "ResNet50": {
        "path": Config.RESNET_MODEL_PATH,
        "type": "resnet50",
    },
    "EfficientNet": {
        "path": Config.EFFICIENTNET_MODEL_PATH,
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
            nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ModelManager:
    """Handles model loading and prediction."""

    def __init__(self):
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
            self.current_model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=False
            )
            num_features = self.current_model.fc.in_features
            self.current_model.fc = nn.Linear(num_features, len(CLASS_NAMES))

        # Load the state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle different state dict formats
        if model_type == "efficientnet":
            # If the state dict has 'model' prefix, it matches our PlantDiseaseModel
            if all(k.startswith("model.") for k in state_dict.keys()):
                self.current_model.load_state_dict(state_dict)
            else:
                # If it doesn't have 'model' prefix, wrap it
                new_state_dict = {"model." + k: v for k, v in state_dict.items()}
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

        try:
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

            # Ensure confidence is a valid number
            confidence_value = float(confidence.item()) * 100
            if not math.isfinite(confidence_value):
                confidence_value = 0.0

            return {
                "label": class_name,
                "confidence": confidence_value,
            }
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                "label": "Error in prediction",
                "confidence": 0.0,
            }


# Initialize model manager
model_manager = ModelManager()


def get_random_image_path():
    """Get a random image path from the dataset."""
    if not os.path.exists(Config.TRAIN_DIR):
        return None

    categories = [
        d for d in os.listdir(Config.TRAIN_DIR)
        if os.path.isdir(os.path.join(Config.TRAIN_DIR, d))
    ]

    if not categories:
        return None

    category = random.choice(categories)
    category_path = os.path.join(Config.TRAIN_DIR, category)

    images = [
        f for f in os.listdir(category_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        return None

    image = random.choice(images)
    return os.path.join(category_path, image)


@app.route("/")
@app.route("/index")
def home():
    models_dir = Config.MODEL_DIR
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
        "model_names": list(available_models.keys()),
        "total_models": len(available_models),
        "model_dir": models_dir,
        "error": (
            None
            if available_models
            else "No trained models found. Please train models first."
        ),
        "class_names": CLASS_NAMES,
    }

    # Get initial random background
    image_path = get_random_image_path()
    if image_path:
        background_url = f'/serve_image/{os.path.relpath(image_path, Config.BASE_DIR).replace(os.sep, "_")}'
        template_data['background_url'] = background_url

    return render_template("index.html", **template_data)


@app.route("/load_model", methods=["POST"])
def load_model():
    try:
        model_name = request.form.get("model_name")
        success = model_manager.load_model(model_name)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Starting prediction process...")
        
        # Get the uploaded file and selected model
        file = request.files['file']
        model_name = request.form['model']
        
        if not file:
            print("No file uploaded")
            return render_template('prediction_result.html',
                                error="No file uploaded")

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"File saved to: {filepath}")
        
        # Get the correct model path
        model_info = models.get(model_name)
        if not model_info:
            print(f"Invalid model selected: {model_name}")
            return render_template('prediction_result.html',
                                error=f"Invalid model selected: {model_name}",
                                image_name=filename)
        
        model_path = model_info['path']
        print(f"Using model path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return render_template('prediction_result.html',
                                error=f"Model file not found at: {model_path}",
                                image_name=filename)
            
        try:
            # Try to load the model
            print(f"Loading model: {model_name}")
            success = model_manager.load_model(model_name)
            if not success:
                print("Failed to load model")
                return render_template('prediction_result.html',
                                    error="Failed to load model",
                                    image_name=filename)
                
            print("Model loaded successfully")
            
            # Read the file for prediction
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
                
            # Make prediction using model manager
            print("Making prediction...")
            prediction_result = model_manager.predict(image_bytes)
            print(f"Prediction result: {prediction_result}")
            
            # Prepare template data
            template_data = {
                'image_name': filename,
                'model_used': model_name,
                'class_name': prediction_result['label'],
                'confidence': prediction_result['confidence'],
                'error': None
            }
            print(f"Rendering template with data: {template_data}")
            
            # Return results
            return render_template('prediction_result.html', **template_data)

        except Exception as e:
            print(f"Error during model loading or prediction: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return render_template('prediction_result.html',
                                error=f"Error processing image: {str(e)}",
                                image_name=filename)

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return render_template('prediction_result.html',
                             error=str(e),
                             image_name=filename if 'filename' in locals() else None)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        predictor = Predictor(model_name="best_model.pth")

        # Get multiple images from request
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files selected"}), 400

        predictions = []
        for file in files:
            temp_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
            )
            file.save(temp_path)

            class_id, confidence = predictor.predict(temp_path)
            predictions.append(
                {
                    "filename": file.filename,
                    "class_id": class_id,
                    "confidence": confidence * 100,
                }
            )

            os.remove(temp_path)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_random_background")
def get_random_background():
    """API endpoint to get a random background image."""
    image_path = get_random_image_path()
    if not image_path:
        return jsonify({"error": "No images found"}), 404

    # Convert the file path to a URL
    relative_path = os.path.relpath(image_path, Config.BASE_DIR)
    background_url = f'/serve_image/{relative_path.replace(os.sep, "_")}'
    return jsonify({"background_url": background_url})


@app.route("/serve_image/<path:filename>")
def serve_image(filename):
    """Serve the image file."""
    # Convert URL-safe path back to system path
    real_path = os.path.join(Config.BASE_DIR, *filename.split("_"))
    return send_file(real_path)


@app.route("/wh")
def wh_page():
    return render_template("wh.html")


@app.route("/contribute", methods=["POST"])
def contribute():
    try:
        disease_label = request.form.get("disease_label")
        if not disease_label:
            return jsonify({'error': 'Please select a disease label'}), 400

        if 'training_images' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('training_images')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Define the dataset path
        dataset_path = os.path.join(Config.TRAIN_DIR, disease_label)
        os.makedirs(dataset_path, exist_ok=True)

        saved_files = []
        for file in files:
            if file and file.filename:
                # Generate a unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                original_extension = os.path.splitext(file.filename)[1]
                new_filename = f"{disease_label}_{timestamp}{original_extension}"

                # Save the file to the dataset directory
                file_path = os.path.join(dataset_path, new_filename)
                file.save(file_path)
                saved_files.append(new_filename)

        return jsonify({
            'success': True,
            'message': (
                f"Successfully contributed {len(saved_files)} images to the {disease_label} dataset. "
                "Remember to retrain the model to include these new images in its learning."
            )
        })

    except Exception as e:
        return jsonify({'error': f'Error contributing images: {str(e)}'}), 500


@app.route("/debug")
def debug():
    try:
        # Initialize verification results
        verify_results = {
            'base_dir': Config.BASE_DIR,
            'model_dir': Config.MODEL_DIR,
            'directories_exist': False
        }

        # Check if critical directories exist
        critical_dirs = {
            'BASE_DIR': Config.BASE_DIR,
            'MODEL_DIR': Config.MODEL_DIR,
            'DATA_DIR': Config.DATA_DIR,
            'STATIC_DIR': Config.STATIC_DIR,
            'CHECKPOINT_DIR': Config.CHECKPOINT_DIR
        }
        
        existing_dirs = {name: os.path.exists(path) for name, path in critical_dirs.items()}
        verify_results['directories_exist'] = all(existing_dirs.values())

        # Get model files status
        model_files = {
            'best_model.pth': Config.BEST_MODEL_PATH,
            'efficientnet_model.pth': Config.EFFICIENTNET_MODEL_PATH,
            'resnet50_model.pth': Config.RESNET_MODEL_PATH
        }
        
        models_status = {}
        for name, path in model_files.items():
            models_status[name] = {
                'exists': os.path.exists(path),
                'path': path,
                'size': f"{os.path.getsize(path) / (1024*1024):.2f} MB" if os.path.exists(path) else "N/A"
            }

        # Get available models count and names
        available_models = [name for name, info in models_status.items() if info['exists']]
        total_models = len(available_models)
        model_names = available_models if available_models else None

        # Debug prints
        print(f"BASE_DIR exists: {os.path.exists(Config.BASE_DIR)} - {Config.BASE_DIR}")
        print(f"MODEL_DIR exists: {os.path.exists(Config.MODEL_DIR)} - {Config.MODEL_DIR}")
        if os.path.exists(Config.MODEL_DIR):
            print(f"Models found: {os.listdir(Config.MODEL_DIR)}")

        # Run test predictions and save images
        test_results = {}
        try:
            # Create directory for test images
            test_images_dir = os.path.join('static', 'test_images')
            os.makedirs(test_images_dir, exist_ok=True)
            
            # Test plant dataset
            plant_image, plant_prediction = get_random_test_prediction("plant")
            if plant_image:
                plant_path = os.path.join(test_images_dir, 'plant_test.jpg')
                plant_image.save(plant_path)
                test_results.update({
                    'plant_image': url_for('static', filename='test_images/plant_test.jpg'),
                    'plant_true_class': plant_prediction['true_class'],
                    'plant_predicted': plant_prediction['predicted_class'],
                    'plant_confidence': plant_prediction['confidence']
                })

            # Test landscape dataset
            landscape_image, landscape_prediction = get_random_test_prediction("landscape")
            if landscape_image:
                landscape_path = os.path.join(test_images_dir, 'landscape_test.jpg')
                landscape_image.save(landscape_path)
                test_results.update({
                    'landscape_image': url_for('static', filename='test_images/landscape_test.jpg'),
                    'landscape_true_class': "Landscape",  # Always "Landscape" for landscape images
                    'landscape_predicted': landscape_prediction['predicted_class'],
                    'landscape_confidence': landscape_prediction['confidence']
                })

        except Exception as e:
            print(f"Error running tests: {str(e)}")

        return render_template('debug.html',
                             verify_results=verify_results,
                             models=models_status,
                             total_models=total_models,
                             model_names=model_names,
                             model_dir=Config.MODEL_DIR,
                             error=None if verify_results['directories_exist'] else "Required directories are missing",
                             test_results=test_results)

    except Exception as e:
        print(f"Debug route error: {str(e)}")
        return render_template('debug.html', error=str(e))


def get_random_test_prediction(dataset_type):
    """Get a random image and its prediction."""
    try:
        # Get random test image
        if dataset_type == "plant":
            test_dir = os.path.join(Config.BASE_DIR, "data", "plantDataset", "test")
        else:
            test_dir = os.path.join(Config.BASE_DIR, "data", "landscapeDataset")
        
        if not os.path.exists(test_dir):
            return None, None

        if dataset_type == "plant":
            # Get random class directory for plant dataset
            class_dirs = [d for d in os.listdir(test_dir) 
                         if os.path.isdir(os.path.join(test_dir, d))]
            if not class_dirs:
                return None, None
            
            class_dir = random.choice(class_dirs)
            class_path = os.path.join(test_dir, class_dir)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            # For landscape dataset, all images are in the root directory
            class_dir = "Not_A_Plant"  # Changed from "Landscape"
            class_path = test_dir
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            return None, None

        image_name = random.choice(images)
        image_path = os.path.join(class_path, image_name)
        
        # Load and predict
        image = Image.open(image_path)
        predictor = Predictor(Config.BEST_MODEL_PATH, "resnet50")
        class_idx, confidence = predictor.predict(image_path)
        
        # Convert class index to name using DEFAULT_CLASS_NAMES
        if class_idx in DEFAULT_CLASS_NAMES:
            predicted_class = f"{DEFAULT_CLASS_NAMES[class_idx][0]} - {DEFAULT_CLASS_NAMES[class_idx][1]}"
        else:
            predicted_class = f"Unknown (Class {class_idx})"

        return image, {
            'true_class': class_dir,
            'predicted_class': predicted_class,
            'confidence': confidence
        }

    except Exception as e:
        print(f"Error getting test prediction: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Create all necessary directories
    ensure_directories()
    
    # Check if model files exist
    for model_name, model_info in models.items():
        if not os.path.exists(model_info["path"]):
            print(f"Warning: Model file not found: {model_info['path']}")

    # Add this after app initialization
    print("\nChecking model files:")
    for model_name, model_info in models.items():
        path = model_info['path']
        exists = os.path.exists(path)
        print(f"Model: {model_name}")
        print(f"Path: {path}")
        print(f"Exists: {exists}")
        if exists:
            print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
        print()

    app.run(debug=True)
