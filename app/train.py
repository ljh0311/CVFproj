import os
import torch
from torch.utils.data import DataLoader
from app.config import Config
from app.data.dataset import CustomImageDataset
from app.data.transforms import DataTransforms
from app.utils.dataset_manager import DatasetManager
from app.models.model import ModelBuilder, Trainer
from app.utils.visualization import plot_training_history


def main():
    """Main training function."""
    print("Starting the training pipeline...")

    # Create necessary directories
    os.makedirs(os.path.join(Config.BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "data"), exist_ok=True)

    # 1. Set up dataset
    dataset_path = os.path.join(Config.BASE_DIR, "data", "plantDataset")
    if not os.path.exists(dataset_path):
        print("\nDataset not found. Attempting to extract from zip file...")
        try:
            dataset_manager = DatasetManager(
                Config.ZIP_PATH, os.path.join(Config.BASE_DIR, "data")
            )
            dataset_manager.setup_dataset()
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            print("Please ensure the dataset zip file is in the correct location:")
            print(f"Expected path: {Config.ZIP_PATH}")
            return
        except Exception as e:
            print(f"\nError extracting dataset: {str(e)}")
            return
    else:
        print(f"\nDataset found at: {dataset_path}")

    # Verify dataset structure
    required_dirs = ["train", "valid", "test"]
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"\nError: Required directory not found: {dir_path}")
            print(
                "Please ensure the dataset is properly extracted with the following structure:"
            )
            print("plantDataset/")
            print("├── train/")
            print("├── valid/")
            print("└── test/")
            return

    # 2. Create data loaders
    try:
        print("\nSetting up data loaders...")
        train_transform, val_transform = DataTransforms.get_transforms()

        # Create the full dataset
        full_dataset = CustomImageDataset(
            os.path.join(Config.BASE_DIR, "data", "plantDataset", "train"),
            transform=train_transform,
        )

        # Store number of classes
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes

        # Create splits while preserving class information
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Add classes attribute to the splits
        train_dataset.classes = class_names
        val_dataset.classes = class_names
        test_dataset.classes = class_names

        print(f"Found {num_classes} classes")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        adjusted_batch_size = max(1, Config.BATCH_SIZE // 4)
        print("adjusted_batch_size: ", adjusted_batch_size)

        # Set number of workers (0 disables parallel loading)
        num_workers_train = 0  # Use 0 workers to reduce CPU usage, adjust as needed
        num_workers_val = 0
        dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=adjusted_batch_size,
                shuffle=True,
                num_workers=num_workers_train,
                pin_memory=True,
            ),
            "val": DataLoader(
                val_dataset, batch_size=adjusted_batch_size, num_workers=num_workers_val
            ),
            "test": DataLoader(
                test_dataset, batch_size=adjusted_batch_size, num_workers=num_workers_val
            ),
        }

    except Exception as e:
        print(f"\nError setting up data loaders: {str(e)}")
        return

    # 3. Create and setup model
    try:
        print("\nInitializing model...")
        model = ModelBuilder.create_model(len(train_dataset.classes))
        model = model.to(Config.DEVICE)
        print(f"Model created with {len(train_dataset.classes)} output classes")
        print(f"Using device: {Config.DEVICE}")

    except Exception as e:
        print(f"\nError creating model: {str(e)}")
        return

    # 4. Train model
    try:
        print("\nStarting training...")
        trainer = Trainer(model, Config.DEVICE)
        history = trainer.train_model(
            dataloaders["train"],
            dataloaders["val"],
            num_epochs=Config.EPOCHS,
            lr=Config.LR,
        )

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return

    # 5. Evaluate on test set
    try:
        print("\nEvaluating on test set...")
        test_loss, test_acc = trainer.evaluate(dataloaders["test"])
        print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")

    # 6. Plot results
    try:
        plot_training_history(*history)
    except Exception as e:
        print(f"\nError plotting results: {str(e)}")

    return model, history


if __name__ == "__main__":
    try:
        model, history = main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
