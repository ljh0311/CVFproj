import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from utils.dataset_manager import DatasetManager


# Configuration
class Config:
    """Configuration parameters for the training pipeline."""
    # Get the base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Hyperparameters
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths with absolute paths
    DATA_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "train")
    TEST_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "test")
    VAL_DIR = os.path.join(BASE_DIR, "data", "plantDataset", "valid")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


class DataTransforms:
    """Handles data transformations and augmentation."""
    @staticmethod
    def get_transforms():
        train_transform = transforms.Compose([
            transforms.Resize(int(Config.IMAGE_SIZE[0] * 1.1)),
            transforms.RandomCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        return train_transform, val_transform

def setup_dataset(zip_path=None, target_dir=None):
    """
    Set up the dataset by extracting and organizing the data files.
    
    Args:
        zip_path (str, optional): Path to the dataset zip file. Defaults to None.
        target_dir (str, optional): Directory to extract dataset to. Defaults to None.
    """
    if zip_path is None:
        zip_path = os.path.join(Config.BASE_DIR, "leafarchive.zip")
    if target_dir is None:
        target_dir = os.path.join(Config.BASE_DIR, "data")
        
    print(f"\nSetting up dataset...")
    print(f"Zip file path: {zip_path}")
    print(f"Target directory: {target_dir}")
    
    dataset_manager = DatasetManager(zip_path, target_dir)
    dataset_manager.setup_dataset()
    
    return dataset_manager.get_class_names()


class CustomImageDataset(Dataset):
    """Custom dataset for loading and processing plant disease images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the image data
            transform (callable, optional): Transform to apply to images. Defaults to None.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get sorted list of class names and create class-to-index mapping
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.images = self._load_images()

    def _load_images(self):
        """
        Load image paths and their corresponding labels.
        
        Returns:
            list: List of tuples containing (image_path, label)
        """
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                images.append((
                    os.path.join(class_path, img_name),
                    self.class_to_idx[class_name]
                ))
        return images

    def __len__(self):
        """Return the total number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and its label at the given index.
        
        Args:
            idx (int): Index of the image to retrieve
            
        Returns:
            tuple: (transformed_image, label)
        """
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ModelBuilder:
    """Handles model creation and initialization."""
    @staticmethod
    def create_model(num_classes):
        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes)
        return model


class Trainer:
    """Handles model training and evaluation."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, train_loader, val_loader, num_epochs=15, lr=0.001, fine_tune_layers=None):
        # Configure trainable parameters
        params_to_train = list(self.model.parameters())[-2:] 
        if fine_tune_layers == "last_two": 
            params_to_train = list(self.model.parameters())[-2:]
        else:
            params_to_train = self.model.parameters()

        optimizer = optim.Adam(params_to_train, lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )

        train_losses, val_losses, val_accuracies = [], [], []
        best_valid_loss = float("inf")

        for epoch in range(num_epochs):
            start_time = time.monotonic()

            # Train the model for one epoch
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Evaluate on validation set
            valid_accuracy, valid_losses = self.evaluate(val_loader)
            valid_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None

            val_losses.append(valid_loss)
            val_accuracies.append(valid_accuracy)

            # Save the model if validation loss improves
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)

            end_time = time.monotonic()

            # Calculate epoch duration
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            # Print epoch statistics
            print(
                f"Epoch: {epoch + 1:02} | Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s"
            )
            print(f"\tTrain Loss: {avg_train_loss:.3f}")
            print(
                f"\tValid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_accuracy * 100:.2f}%"
            )

            # After validation in your epoch loop
            scheduler.step(valid_loss)

        return train_losses, val_losses, val_accuracies

    def evaluate(self, dataloader):
        """
        Evaluate the model on the given dataloader.

        Parameters:
            dataloader (torch.utils.data.DataLoader): The dataloader for evaluation.

        Returns:
            tuple: Validation accuracy and list of average losses over batches.
        """
        self.model.eval()  # Set model to evaluation mode
        datasize = 0  # Total number of samples evaluated
        accuracy = 0  # Total accuracy over batches
        total_loss = 0  # Cumulative loss over batches
        losses = []  # Store losses for each batch

        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to device

                outputs = self.model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Predicted class indices

                if self.criterion is not None:
                    loss = self.criterion(outputs, labels)  # Compute batch loss
                    total_loss += loss.item()
                    losses.append(loss.item())

                # Update accuracy
                accuracy += torch.sum(preds == labels).item()
                datasize += labels.size(0)

        # Compute overall accuracy and average loss
        accuracy /= datasize
        avg_loss = total_loss / len(dataloader) if self.criterion else None

        return accuracy, losses


def main():
    """
    Main function to orchestrate the training process.
    """
    print("Starting the training pipeline...")
    
    # 1. Set up dataset structure
    try:
        class_names = setup_dataset()
        print(f"\nFound {len(class_names)} classes: {class_names}")
    except FileNotFoundError as e:
        print(f"\nWarning: {str(e)}")
        print("Proceeding with existing dataset structure...")
    except Exception as e:
        print(f"\nError setting up dataset: {str(e)}")
        print("Please ensure the dataset zip file is in the correct location.")
        return
    
    # 2. Set up data loaders
    print("\nSetting up datasets and dataloaders...")
    try:
        train_dataset = CustomImageDataset(Config.DATA_DIR, transform=DataTransforms.get_transforms()[0])
        val_dataset = CustomImageDataset(Config.VAL_DIR, transform=DataTransforms.get_transforms()[1])
        test_dataset = CustomImageDataset(Config.TEST_DIR, transform=DataTransforms.get_transforms()[1])
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0),
            "val": DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0),
            "test": DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        }
        
        print(f"Dataset sizes:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error setting up dataloaders: {str(e)}")
        return
    
    # 3. Create and setup model
    print("\nInitializing model...")
    try:
        num_classes = len(train_dataset.classes)
        model = ModelBuilder.create_model(num_classes)
        model = model.to(Config.DEVICE)
        print(f"Model created with {num_classes} classes")
        print(f"Using device: {Config.DEVICE}")
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return
    
    # 4. Train the model
    print("\nStarting training...")
    try:
        trainer = Trainer(model, Config.DEVICE)
        train_losses, val_losses, val_accuracies = trainer.train_model(
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            num_epochs=Config.EPOCHS,
            lr=Config.LR,
            fine_tune_layers=None
        )
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    # 5. Evaluate on test set
    print("\nEvaluating model on test set...")
    try:
        test_accuracy, test_losses = trainer.evaluate(dataloaders["test"])
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    
    # 6. Plot training history
    try:
        plot_training_history(train_losses, val_losses, val_accuracies)
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")
    
    print("\nTraining pipeline completed!")
    return model, (train_losses, val_losses, val_accuracies)

def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    Plot training history including losses and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in val_accuracies], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(os.path.join(Config.BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "models"), exist_ok=True)
    
    try:
        model, history = main()
        print("Program completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
