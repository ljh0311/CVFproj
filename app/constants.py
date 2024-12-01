import os
from app.config import Config

MODEL_DIR = os.path.join(Config.BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, "temp", "uploads")
CHECKPOINT_PATH = os.path.join(Config.BASE_DIR, "checkpoints", "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(Config.BASE_DIR, "checkpoints", "best_model.pth")
 