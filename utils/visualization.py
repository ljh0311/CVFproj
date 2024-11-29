import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_training_history(train_losses, val_losses, val_accuracies, model_name):
    """Plot and save training history."""
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join('static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plots
    plot_filename = f'training_loss_{model_name.lower()}_{timestamp}.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()
    
    return plot_path 
    print(f"\nTraining plots saved to: {loss_plot_path}")
    
    return loss_plot_path 