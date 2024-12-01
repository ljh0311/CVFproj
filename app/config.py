import os
import torch
from flask import Blueprint

class Config:
    """Configuration parameters for the training pipeline."""
    # Base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directory paths
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    
    # Dataset paths
    DATASET_DIR = os.path.join(DATA_DIR, "plantDataset")
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    VAL_DIR = os.path.join(DATASET_DIR, "valid")
    
    # Model paths
    RESNET_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_model.pth")
    EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_model.pth")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    
    # Static paths
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
    PLOTS_FOLDER = os.path.join(STATIC_DIR, "plots")
    
    # Checkpoint paths
    LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    
    # Archive paths
    PLANT_ZIP_PATH = os.path.join(DATA_DIR, "leafarchive.zip")
    LANDSCAPE_ZIP_PATH = os.path.join(DATA_DIR, "landscapePhotos.zip")
    
    # Hyperparameters
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    SECRET_KEY = 'your_secret_key'
    DEBUG = True

    @classmethod
    def update_params(cls, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)