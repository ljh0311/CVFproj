import os
import torch

class Config:
    """Configuration parameters for the training pipeline."""
    # Base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Hyperparameters
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    DATA_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "train")
    TEST_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "test")
    VAL_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "valid")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
    ZIP_PATH = os.path.join(BASE_DIR, "data", "leafarchive.zip")
    CHECKPOINT_PATH = "checkpoints/last_checkpoint.pth"
    BEST_MODEL_PATH = "checkpoints/best_model.pth"

    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225] 