import os

# Use absolute path for project root
BASE_DIR = r"C:\Users\user\Documents\SITstuffs\CompVis\CVFproj"
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Checkpoint paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "1stmodel.pth")

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "test")
VAL_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "valid")

# Match the exact filenames from your models directory
MODEL_FILES = {
    "ResNet50": {
        "filename": "resnet50_model.pth",
        "type": "resnet50"
    },
    "EfficientNet": {
        "filename": "efficientnet_model.pth",
        "type": "efficientnet"
    }
}

# Create necessary directories
for directory in [MODEL_DIR, UPLOAD_FOLDER, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Debug print
print(f"\n=== Constants Initialization ===")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")

# Check model files
print("\n=== Model Files ===")
for model_name, file_info in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, file_info["filename"])
    exists = os.path.exists(path)
    if exists:
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"✓ {model_name}: {file_info['filename']} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {model_name}: {file_info['filename']} (not found)")

# Check checkpoint files
print("\n=== Checkpoint Files ===")
for path in [CHECKPOINT_PATH, BEST_MODEL_PATH]:
    exists = os.path.exists(path)
    if exists:
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"✓ {os.path.basename(path)} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {os.path.basename(path)} (not found)")
 