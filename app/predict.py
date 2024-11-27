import torch
from PIL import Image
from app.config import Config
from app.data.transforms import DataTransforms
from app.models.model import ModelBuilder

class Predictor:
    def __init__(self, model_path, num_classes):
        self.device = Config.DEVICE
        self.model = ModelBuilder.create_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        _, self.transform = DataTransforms.get_transforms()

    def predict(self, image_path):
        """Make a prediction on a single image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item()

def main():
    """Main prediction function."""
    predictor = Predictor(Config.MODEL_SAVE_PATH, num_classes=38)  # Update num_classes as needed
    
    # Example usage
    image_path = "path/to/your/test/image.jpg"
    predicted_class, confidence = predictor.predict(image_path)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()