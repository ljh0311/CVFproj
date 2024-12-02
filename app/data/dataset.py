from torch.utils.data import Dataset
from PIL import Image
import os

from app.predict import DEFAULT_CLASS_NAMES


class CustomImageDataset(Dataset):
    """Custom dataset for loading and processing plant disease images."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get all image files and class names from directory structure
        self.classes = sorted(os.listdir(root_dir))  # Get class names from folders
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        # Collect all images and their labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
