import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import torch


def plot_model_performance(confusion_mat, class_names, save_path=None):
    """
    Plot and save confusion matrix and model performance metrics.

    Args:
        confusion_mat: numpy array of confusion matrix
        class_names: dictionary of class names {index: (plant, condition)}
        save_path: path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Create class labels for the plot
    labels = [f"{plant}-{condition}" for plant, condition in class_names.values()]

    # Plot confusion matrix
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(train_losses, val_losses, val_accuracies, model_name, save_dir='static/plots'):
    """Plot and save training metrics."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_confusion_matrix(model, test_loader, class_names, device, save_dir='static/plots'):
    """Generate and save confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Collect predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path
