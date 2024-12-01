import os
import torch
from torch.utils.data import DataLoader
from app.config import Config
from app.data.dataset import CustomImageDataset
from app.data.transforms import DataTransforms
from app.utils.dataset_manager import DatasetManager
from app.models.model import ModelBuilder, Trainer
from utils.visualization import plot_training_history


def get_next_version(model_type):
    """Get the next available version number for the model."""
    model_dir = os.path.join(Config.BASE_DIR, "models")
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(model_type)]
    if not existing_models:
        return "1.0"
    
    versions = [float(f.split('_v')[1].split('.pth')[0]) for f in existing_models]
    return f"{max(versions) + 1.0:.1f}"


def get_model_filename(model_type, num_classes, epochs, lr, batch_size):
    """Generate a standardized model filename."""
    return f"{model_type}_{num_classes}_{epochs}_{lr}_{batch_size}.pth"


def main(epochs=None, lr=None, batch_size=None, model_type=None, model_name=None, progress_callback=None):
    """Main training function."""
    # Use provided parameters or defaults from Config
    epochs = epochs or Config.EPOCHS
    lr = lr or Config.LR
    batch_size = batch_size or Config.BATCH_SIZE
    model_type = model_type or "resnet50"
    
    print("Starting the training pipeline...")
    print(f"\nTraining parameters:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Learning Rate: {lr}")
    
    # Create necessary directories
    os.makedirs(os.path.join(Config.BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "data"), exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 1. Set up dataset
    dataset_path = os.path.join(Config.BASE_DIR, "data", "plantDataset")
    dataset_manager = DatasetManager(Config.ZIP_PATH, os.path.join(Config.BASE_DIR, "data"))
    
    if not os.path.exists(dataset_path):
        print("\nDataset not found. Attempting to extract from zip file...")
        try:
            dataset_manager = DatasetManager(
                Config.PLANT_ZIP_PATH, os.path.join(Config.BASE_DIR, "data")
            )
            dataset_manager.setup_dataset(dataset_type="plant")
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            print("Please ensure the dataset zip file is in the correct location:")
            print(f"Expected path: {Config.PLANT_ZIP_PATH}")
            return
        except Exception as e:
            print(f"\nError extracting dataset: {str(e)}")
            return
    else:
        print(f"\nDataset found at: {dataset_path}")

    # 2. Create data loaders
    try:
        print("\nSetting up data loaders...")
        train_transform, val_transform = DataTransforms.get_transforms()

        # Create the full dataset
        full_dataset = CustomImageDataset(
            os.path.join(Config.BASE_DIR, "data", "plantDataset", "train"),
            transform=train_transform,
        )

        # Store number of classes and save class names
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes

        # Save class names to a file
        class_names_path = os.path.join(Config.BASE_DIR, "models", "class_names.txt")
        with open(class_names_path, 'w', encoding='utf-8') as f:
            for idx, class_name in enumerate(class_names):
                f.write(f"{idx}:{class_name}\n")

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
        
        # Get batch size from config, with a minimum of 1
        batch_size = max(1, Config.BATCH_SIZE)
        print(f"Batch size: {batch_size}")

        dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            ),
            "val": DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                num_workers=0
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=0,
            ),
        }

    except Exception as e:
        print(f"\nError setting up data loaders: {str(e)}")
        return

    # 3. Create and setup model
    try:
        print("\nInitializing model...")
        num_classes = len(train_dataset.classes)
        model = ModelBuilder.create_model(num_classes)
        model_type = model_type or model.__class__.__name__.lower()
        
        # Generate model filename with parameters
        model_filename = get_model_filename(model_type, num_classes, epochs, lr, batch_size)
        checkpoint_filename = f"checkpoint_{model_filename}"
        
        # Update paths with new filenames
        model_save_path = os.path.join(Config.MODEL_DIR, model_filename)
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)
        
        # Check for existing checkpoint with same parameters
        if os.path.exists(checkpoint_path):
            print(f"\nFound existing checkpoint: {checkpoint_filename}")
            try:
                checkpoint = torch.load(checkpoint_path)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    print("Loading checkpoint...")
                    model.load_state_dict(checkpoint['state_dict'])
                    print("Checkpoint loaded successfully!")
            except Exception as e:
                print(f"Warning: Error loading checkpoint ({str(e)}). Starting fresh training...")
        
        model = model.to(Config.DEVICE)
        print(f"Model: {model_filename}")
        print(f"Number of classes: {num_classes}")
        print(f"Device: {Config.DEVICE}")

    except Exception as e:
        print(f"\nError creating model: {str(e)}")
        return

    # 4. Train model
    try:
        print("\nStarting training...")
        print(f"Training parameters:")
        print(f"- Epochs: {Config.EPOCHS}")
        print(f"- Learning Rate: {Config.LR}")
        print(f"- Batch Size: {batch_size}")
        
        trainer = Trainer(model, Config.DEVICE)
        trainer.model_save_path = model_save_path
        trainer.checkpoint_path = checkpoint_path
        history = trainer.train_model(
            dataloaders["train"],
            dataloaders["val"],
            num_epochs=epochs,
            lr=lr,
        )

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return

    # 5. Evaluate on test set
    try:
        print("\nEvaluating on test set...")
        test_loss, test_acc = trainer.evaluate(dataloaders["test"])
        print(f"Final Test Accuracy: {test_acc:.2f}%")

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")

    # 6. Plot and save results
    try:
        print("\nGenerating and saving training plots...")
        plot_path = plot_training_history(
            train_losses=history[0],
            val_losses=history[1],
            val_accuracies=history[2],
            model_name=model_filename  # Use versioned name
        )
        print(f"Training plots saved to: {plot_path}")
    except Exception as e:
        print(f"\nError plotting and saving results: {str(e)}")

    return model, history


if __name__ == "__main__":
    try:
        # Get user input for training parameters
        epochs = int(input("Enter number of epochs (default 10): ") or 10)
        batch_size = int(input("Enter batch size (default 32): ") or 32)
        lr = float(input("Enter learning rate (default 0.001): ") or 0.001)
        
        # Update config with user parameters
        Config.update_params(
            EPOCHS=epochs,
            BATCH_SIZE=batch_size,
            LR=lr
        )
        
        # Call main without parameters
        model, history = main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
