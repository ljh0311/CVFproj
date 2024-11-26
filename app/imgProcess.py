from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import numpy as np

def preprocess_images(data_dir, target_size=(224, 224)):
    """
    Preprocess images from a directory structure where images are grouped by class.
    
    Args:
        data_dir (str): Path to the dataset directory.
        target_size (tuple): Desired size for the images (height, width).
        
    Returns:
        np.ndarray: Array of preprocessed images.
        np.ndarray: Array of one-hot encoded labels.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Sort to maintain consistent class order
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip if it's not a directory
        
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                # Load and resize image
                img = Image.open(file_path).convert("RGB")  # Convert to RGB if not already
                img = img.resize(target_size)
                images.append(np.array(img) / 255.0)  # Normalize pixel values to [0, 1]
                labels.append(class_to_index[class_name])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=len(class_names))
    return images, labels
