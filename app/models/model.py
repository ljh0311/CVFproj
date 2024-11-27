import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os

from app.config import Config


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
    def create_model(num_classes, pretrained=True, model_type="standard"):
        """Create and initialize the model."""
        if model_type == "standard":
            model = models.resnet50(pretrained=pretrained)
        elif model_type == "fast":
            model = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modify the final layer for our number of classes
        if model_type == "standard":
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        else:
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)

        return model


class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.best_val_loss = float("inf")

    def save_checkpoint(self, epoch, optimizer, scheduler, train_losses, val_losses, val_accuracies, is_best=False):
        """Save training checkpoint and optionally save as best model."""
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
        
        # Save regular checkpoint
        torch.save(checkpoint, Config.CHECKPOINT_PATH)
        
        # If this is the best model, save a copy
        if is_best:
            torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
    
    def load_checkpoint(self, optimizer=None, scheduler=None):
        """Load a training checkpoint."""
        if not os.path.exists(Config.CHECKPOINT_PATH):
            return None
            
        checkpoint = torch.load(Config.CHECKPOINT_PATH)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1
        )

        # Try to load checkpoint if exists
        checkpoint = self.load_checkpoint(optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0
        train_losses = checkpoint.get('train_losses', []) if checkpoint else []
        val_losses = checkpoint.get('val_losses', []) if checkpoint else []
        val_accuracies = checkpoint.get('val_accuracies', []) if checkpoint else []

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            # Create progress bar for training
            train_pbar = tqdm(train_loader, desc='Training', 
                            leave=True, 
                            unit='batch')
            
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Update progress bar description with current loss
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc='Validation', 
                          leave=True, 
                          unit='batch')
            
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar description
                    current_acc = 100. * correct / total
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.2f}%'
                    })

            val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save checkpoint after each epoch
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

            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"Training Loss: {epoch_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            print("-" * 60)

        return train_losses, val_losses, val_accuracies

    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss
