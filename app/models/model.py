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
        self.model_save_path = None
        self.checkpoint_path = None
        self.best_val_loss = float("inf")
        os.makedirs('checkpoints', exist_ok=True)

    def save_checkpoint(self, epoch, optimizer, scheduler, train_losses, val_losses, val_accuracies, is_best=False):
        """Save training checkpoint and optionally save as best model."""
        # Create checkpoints directory if it doesn't exist
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        if is_best:
            torch.save(self.model.state_dict(), BEST_MODEL_PATH)
    
    def load_checkpoint(self, optimizer=None, scheduler=None):
        """Load a training checkpoint."""
        if not os.path.exists(CHECKPOINT_PATH):
            return None
            
        checkpoint = torch.load(CHECKPOINT_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint

    def train_model(
        self, train_loader, val_loader, num_epochs=10, lr=0.001, fine_tune_layers=None
    ):
        """Train the model."""
        criterion = nn.CrossEntropyLoss()
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

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save checkpoint after each epoch
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Add this before saving checkpoints
            os.makedirs('checkpoints', exist_ok=True)
            
            self.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                val_accuracies=val_accuracies,
                is_best=is_best
            )

            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"Training Loss: {epoch_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            print("-" * 60)

        # Save final model
        self.save_model()

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

    def save_model(self):
        """Save the final model."""
        if self.model_save_path:
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Model saved to: {self.model_save_path}")
