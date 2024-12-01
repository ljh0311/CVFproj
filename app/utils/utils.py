import os
from app.config import Config

def get_next_version(model_type):
    """
    Get the next available version number for a given model type.
    
    Args:
        model_type (str): The type/name of the model
        
    Returns:
        int: The next available version number
    """
    models_dir = os.path.join(Config.BASE_DIR, "models")
    existing_versions = []
    
    # Check existing model files
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.startswith(model_type) and "_v" in filename:
                try:
                    version = int(filename.split("_v")[-1].split(".")[0])
                    existing_versions.append(version)
                except ValueError:
                    continue
    
    # Return next version (1 if no existing versions found)
    return max(existing_versions, default=0) + 1 