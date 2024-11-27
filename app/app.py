from flask import Flask, request, render_template, jsonify, send_file
import torch
from PIL import Image
import io
import os
from imgProcess import CustomImageDataset
from torchvision import transforms
from main import Config, ModelBuilder
import random

app = Flask(__name__)

# Update the paths to be relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize models dictionary with absolute paths
models = {
    "resnet_standard": {
        "path": os.path.join(BASE_DIR, "models", "best_model.pth"),
        "type": "standard"
    },
    "resnet_fast": {
        "path": os.path.join(BASE_DIR, "models", "best_model_fast.pth"),
        "type": "fast"
    }
}

class ModelManager:
    """Handles model loading and prediction."""
    def __init__(self):
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])
        
        # Update data directory path to be absolute
        self.data_dir = os.path.join(BASE_DIR, "data", "plantDataset", "train")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory not found at {self.data_dir}")
            print("Creating sample classes for testing...")
            # Create a temporary structure with sample classes
            self.classes = ["Healthy", "Disease_1", "Disease_2"]
        else:
            # Load class names from actual dataset
            self.dataset = CustomImageDataset(self.data_dir)
            self.classes = self.dataset.classes

    def load_model(self, model_name):
        """Load a specific model."""
        if model_name not in models:
            raise ValueError(f"Invalid model name: {model_name}")
            
        model_config = models[model_name]
        model_path = model_config["path"]
        model_type = model_config["type"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Create and load model
        self.current_model = ModelBuilder.create_model(
            len(self.classes), 
            pretrained=False,
            model_type=model_type
        )
        self.current_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.current_model.eval()
        return True

    def predict(self, image_bytes):
        """Make prediction on an image."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
            
        # Prepare image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = self.current_model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return {
            'label': self.classes[predicted.item()],
            'confidence': float(confidence.item()) * 100
        }

# Initialize model manager
model_manager = ModelManager()

def get_random_image_path():
    """Get a random image path from the dataset."""
    dataset_path = os.path.join(BASE_DIR, "data", "New Plant Diseases Dataset", "train")
    
    # Get all subdirectories (disease categories)
    categories = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not categories:
        return None
        
    # Select random category
    category = random.choice(categories)
    category_path = os.path.join(dataset_path, category)
    
    # Get all images in the category
    images = [f for f in os.listdir(category_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        return None
        
    # Select random image
    image = random.choice(images)
    return os.path.join(category_path, image)

@app.route('/')
def home():
    # Check if models directory exists
    models_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return render_template('index.html', 
                             error="No trained models found. Please train models first.",
                             model_names=[])
    
    available_models = {name: path for name, path in models.items() 
                       if os.path.exists(path)}
    
    if not available_models:
        return render_template('index.html',
                             error="No trained models found. Please train models first.",
                             model_names=[])
    
    # Get initial random background
    image_path = get_random_image_path()
    background_url = f'/serve_image/{os.path.relpath(image_path, BASE_DIR).replace(os.sep, "_")}' if image_path else None
    
    return render_template('index.html', model_names=available_models.keys(), background_url=background_url)

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        model_name = request.form.get('model_name')
        success = model_manager.load_model(model_name)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', 
                             error="No file uploaded",
                             model_names=models.keys())
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html',
                             error="No file selected",
                             model_names=models.keys())
    
    # Load selected model
    model_name = request.form.get('model')
    try:
        model_manager.load_model(model_name)
    except Exception as e:
        return render_template('index.html', 
                             error=f"Failed to load model: {str(e)}",
                             model_names=models.keys())
    
    # Make prediction
    image_bytes = file.read()
    prediction = model_manager.predict(image_bytes)
    return jsonify({'prediction': prediction})

@app.route('/get_random_background')
def get_random_background():
    """API endpoint to get a random background image."""
    image_path = get_random_image_path()
    if not image_path:
        return jsonify({'error': 'No images found'}), 404
        
    # Convert the file path to a URL
    relative_path = os.path.relpath(image_path, BASE_DIR)
    background_url = f'/serve_image/{relative_path.replace(os.sep, "_")}'
    return jsonify({'background_url': background_url})

@app.route('/serve_image/<path:filename>')
def serve_image(filename):
    """Serve the image file."""
    # Convert URL-safe path back to system path
    real_path = os.path.join(BASE_DIR, *filename.split('_'))
    return send_file(real_path)

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    
    print(f"Project directory: {BASE_DIR}")
    print(f"Models directory: {os.path.join(BASE_DIR, 'models')}")
    print(f"Data directory: {os.path.join(BASE_DIR, 'data')}")
    
    app.run(debug=True)
