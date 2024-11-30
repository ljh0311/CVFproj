import matplotlib.pyplot as plt
import seaborn as sns
import os


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


def plot_training_history(
    train_losses, val_losses, val_accuracies, model_name, save_path=None
):
    """Plot training history including losses and validation accuracy."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs, [acc * 100 for acc in val_accuracies], "g-", label="Validation Accuracy"
    )
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
