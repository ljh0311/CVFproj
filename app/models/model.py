import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import os
from app.config import Config
from app.utils import get_next_version

from app.constants import CHECKPOINT_PATH, BEST_MODEL_PATH

import time


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
            # Initialize without pretrained weights
            model = models.resnet50(weights=None)
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

    def __init__(self, model, device, progress_callback=None):
        self.model = model
        self.device = device
        self.model_save_path = None
        self.checkpoint_path = None
        self.best_val_loss = float("inf")
        self.criterion = nn.CrossEntropyLoss()
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
            'best_val_loss': self.best_val_loss,
            'model_type': self.model.__class__.__name__  # Add model type to checkpoint
        }
        
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        if is_best:
            torch.save(self.model.state_dict(), BEST_MODEL_PATH)
    
    def load_checkpoint(self, optimizer=None, scheduler=None):
        """Load a training checkpoint."""
        if not os.path.exists(CHECKPOINT_PATH):
            return None
            
        checkpoint = torch.load(CHECKPOINT_PATH)
        
        # Skip loading the fc layer weights if dimensions don't match
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k not in ['fc.weight', 'fc.bias']}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint

    def train_model(self, train_loader, val_loader, num_epochs=10, lr=0.001, progress_callback=None):
        """Train the model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        # Calculate total steps for progress bar
        total_steps = len(train_loader) * num_epochs
        start_time = time.time()

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            # Create progress bar for this epoch
            pbar = tqdm(enumerate(train_loader), 
                       total=len(train_loader),
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       ncols=100,
                       leave=True)
            
            for batch_idx, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate progress and time estimates
                steps_done = epoch * len(train_loader) + batch_idx + 1
                time_elapsed = time.time() - start_time
                time_per_step = time_elapsed / steps_done
                steps_remaining = total_steps - steps_done
                eta = steps_remaining * time_per_step
                
                # Format time strings
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
                
                # Calculate hours, minutes, seconds remaining
                hours_remaining = int(eta // 3600)
                minutes_remaining = int((eta % 3600) // 60)
                seconds_remaining = int(eta % 60)
                remaining_str = f"{hours_remaining}h {minutes_remaining}m {seconds_remaining}s"
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ETA': eta_str,
                    'remaining': remaining_str,
                    'elapsed': elapsed_str
                })
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        epoch=epoch + 1,
                        total_epochs=num_epochs,
                        batch=batch_idx + 1,
                        total_batches=len(train_loader),
                        loss=loss.item(),
                        eta=eta,
                        time_remaining=remaining_str
                    )

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            print("-" * 60)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                val_accuracies=val_accuracies,
                is_best=is_best
            )

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
        accuracy = (correct / total) * 100  # Fixed accuracy calculation
        return avg_loss, accuracy

    def save_model(self):
        """Save the final model."""
        if self.model_save_path:
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Model saved to: {self.model_save_path}")
