import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import kagglehub
import shutil

# Create data directory if it doesn't exist
os.makedirs('/data', exist_ok=True)

# Download dataset
if not os.path.exists("/data/plantDataset"):
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    # Define source and destination paths
    source_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)")
    dest_dir = "/data/plantDataset"

# Remove destination directory if it already exists
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# Copy and rename the directory
shutil.copytree(source_dir, dest_dir)

print(f"Dataset downloaded and extracted to: {dest_dir}")

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.samples.append((file_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
