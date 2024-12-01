import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import os
from app.config import Config
from app.utils import get_next_version

from app.constants import CHECKPOINT_PATH, BEST_MODEL_PATH


class ModelBuilder:
    """Handles model creation and initialization."""

    @staticmethod
    def load_trained_model(model_path, num_classes, device):
        """Load a trained model from path."""
        model = ModelBuilder.create_model(num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        return model

    @staticmethod
    def create_model(num_classes, model_type="resnet50", pretrained=True):
        """Create a new model instance."""
        print(f"Creating {model_type} model with {num_classes} classes...")
        
        if model_type == "resnet50":
            # Initialize with pretrained weights if specified
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            # Modify the final fully connected layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        
        elif model_type == "efficientnet":
            model = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model


class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, train_loader, val_loader, num_epochs=10, lr=0.001, progress_callback=None):
        """
        Train the model with optional progress callback
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        epoch=epoch + 1,
                        total_epochs=num_epochs,
                        batch=batch_idx + 1,
                        total_batches=len(train_loader),
                        loss=loss.item()
                    )

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Call progress callback with validation results
            if progress_callback:
                progress_callback(
                    epoch=epoch + 1,
                    total_epochs=num_epochs,
                    val_loss=val_loss,
                    val_acc=val_acc
                )

            # Save checkpoint if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'version': get_next_version(self.model.__class__.__name__)
                    }
                    torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, 'last_checkpoint.pth'))
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {str(e)}")

        return train_losses, val_losses, val_accuracies

    def evaluate(self, loader):
        """
        Evaluate the model on the given loader
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
