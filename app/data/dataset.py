from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """Custom dataset for loading and processing plant disease images."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        print(f"Loading dataset from: {data_dir}")
        
        # Get all image files from subdirectories
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))  # Each subdirectory is a class
        
        # print(f"Found classes: {self.classes}")
        
        valid_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):  # Make sure it's a directory
                for img_name in os.listdir(class_path):
                    if img_name.endswith(valid_extensions):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 