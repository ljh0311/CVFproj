import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime


def plot_training_history(
    train_losses,
    val_losses,
    val_accuracies,
    model_name,
    y_true=None,
    y_pred=None,
    class_names=None,
):
    """Plot and save training history with additional performance metrics."""
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join("static", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.title(f"{model_name} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # 3. Confusion Matrix (if prediction data is provided)
    if y_true is not None and y_pred is not None:
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
        )
        plt.title(f"{model_name} - Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    # 4. Performance Metrics (if prediction data is provided)
    if y_true is not None and y_pred is not None:
        plt.subplot(2, 2, 4)
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names if class_names else None,
            output_dict=True,
        )

        # Extract metrics
        metrics_df = {"Precision": [], "Recall": [], "F1-Score": []}

        for label in report.keys():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                metrics_df["Precision"].append(report[label]["precision"])
                metrics_df["Recall"].append(report[label]["recall"])
                metrics_df["F1-Score"].append(report[label]["f1-score"])

        x = np.arange(len(metrics_df["Precision"]))
        width = 0.25

        plt.bar(
            x - width,
            metrics_df["Precision"],
            width,
            label="Precision",
            color="skyblue",
        )
        plt.bar(x, metrics_df["Recall"], width, label="Recall", color="lightgreen")
        plt.bar(
            x + width, metrics_df["F1-Score"], width, label="F1-Score", color="salmon"
        )

        plt.title(f"{model_name} - Performance Metrics by Class")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.xticks(
            x, class_names if class_names else range(len(x)), rotation=45, ha="right"
        )
        plt.legend()
        plt.grid(True, axis="y")

    # Adjust layout and save
    plt.tight_layout()

    # Save plots
    plot_filename = f"model_performance_{model_name.lower()}_{timestamp}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close the figure to free memory
    plt.close()

    return plot_path


def generate_performance_report(y_true, y_pred, class_names=None):
    """Generate a detailed performance report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names if class_names else None,
        output_dict=True,
    )

    # Convert to percentage and round to 2 decimal places
    for key in report:
        if isinstance(report[key], dict):
            for metric in report[key]:
                report[key][metric] = round(report[key][metric] * 100, 2)

    return report
