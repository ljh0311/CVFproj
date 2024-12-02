import os
import shutil
import zipfile
import sys

from app.config import Config


class DatasetManager:
    """Handles dataset download, extraction and preparation."""

    def __init__(self, zip_path, target_dir):
        self.zip_path = os.path.abspath(zip_path)
        self.target_dir = os.path.abspath(target_dir)

    def setup_dataset(self, dataset_type="plant"):
        """Extract and organize the dataset."""
        print("\nSetting up dataset...")
        
        # Use correct zip path based on dataset type
        if dataset_type == "plant":
            self.zip_path = Config.PLANT_ZIP_PATH
        elif dataset_type == "landscape":
            self.zip_path = Config.LANDSCAPE_ZIP_PATH
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        print(f"Using zip file: {self.zip_path}")
        print(f"Target directory: {self.target_dir}")

        # Verify the zip file exists and is valid
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Dataset zip file not found at: {self.zip_path}")

        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"Provided file is not a valid zip file: {self.zip_path}")

        # Create target directory if it doesn't exist
        os.makedirs(self.target_dir, exist_ok=True)

        # Determine dataset type based on zip file name
        zip_name = os.path.basename(self.zip_path).lower()
        is_plant_dataset = "plant" in zip_name
        is_landscape_dataset = "landscape" in zip_name

        if not (is_plant_dataset or is_landscape_dataset):
            raise ValueError("Unrecognized dataset type. Zip file name should contain 'plant' or 'landscape'")

        try:
            if is_landscape_dataset:
                # For landscape dataset, extract directly to target directory
                print("Setting up landscape dataset...")
                landscape_dir = os.path.join(self.target_dir, "landscapeDataset")
                os.makedirs(landscape_dir, exist_ok=True)
                
                with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                    zip_ref.extractall(landscape_dir)
                print("Landscape dataset extraction completed!")
                
            else:
                # For plant dataset, handle the nested structure
                print("Setting up plant disease dataset...")
                dataset_path = os.path.join(self.target_dir, "plantDataset")
                temp_dir = os.path.join(self.target_dir, "temp_extract")

                # Clean up existing directories
                for path in [dataset_path, temp_dir]:
                    if os.path.exists(path):
                        print(f"Removing existing directory: {path}")
                        shutil.rmtree(path)

                # Create fresh directories
                os.makedirs(temp_dir)
                os.makedirs(dataset_path)

                # Extract the zip file
                print(f"Extracting dataset from {self.zip_path}...")
                with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
                print("Extraction completed!")

                # Find the source directories
                base_path = os.path.join(temp_dir, "New Plant Diseases Dataset(Augmented)")
                if os.path.exists(os.path.join(base_path, "New Plant Diseases Dataset(Augmented)")):
                    base_path = os.path.join(base_path, "New Plant Diseases Dataset(Augmented)")

                # Move directories
                for dir_name in ["train", "valid", "test"]:
                    src = os.path.join(base_path, dir_name)
                    dst = os.path.join(dataset_path, dir_name)

                    if os.path.exists(src):
                        print(f"Moving {dir_name} directory...")
                        shutil.copytree(src, dst)
                    else:
                        print(f"Warning: {dir_name} directory not found in source")

                # Clean up
                print("Cleaning up temporary files...")
                shutil.rmtree(temp_dir)

                # Verify the final structure
                self._verify_dataset_structure(dataset_path)
                print("Dataset setup completed successfully!")

        except Exception as e:
            print(f"\nError processing landscape dataset: {str(e)}")
            # Clean up on error
            if is_plant_dataset:
                for path in [temp_dir, dataset_path]:
                    if os.path.exists(path):
                        shutil.rmtree(path)
            raise

    def _verify_dataset_structure(self, dataset_path):
        """Verify dataset structure."""
        required_dirs = ["train", "valid", "test"]

        print("\nVerifying dataset structure...")
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

            class_dirs = [
                d
                for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d))
            ]

            if not class_dirs:
                raise ValueError(f"No class directories found in {dir_path}")

            print(f"Found {len(class_dirs)} classes in {dir_name} directory")

    def _print_directory_structure(self, startpath):
        """Print directory structure for debugging."""
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files[:5]:
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... ({len(files)-5} more files)")

    def get_class_names(self):
        """Get list of class names."""
        train_dir = os.path.join(self.target_dir, "plantDataset", "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found at: {train_dir}")

        class_names = [
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        return sorted(class_names)

    def rename_images(self, dataset_name="landscapeDataset", prefix="landscape"):
        """Rename all images in the dataset with a specified prefix and number."""
        dataset_path = os.path.join(self.target_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        print(f"\nRenaming images with prefix: {prefix}_")
        image_counter = 1

        for split in ["train", "valid", "test"]:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                print(f"Processing {split}/{class_name}...")
                for old_name in os.listdir(class_path):
                    if not any(old_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        continue

                    # Get file extension
                    ext = os.path.splitext(old_name)[1]
                    
                    # Create new name
                    new_name = f"{prefix}_{image_counter}{ext}"
                    old_path = os.path.join(class_path, old_name)
                    new_path = os.path.join(class_path, new_name)

                    # Rename file
                    os.rename(old_path, new_path)
                    image_counter += 1

        print(f"Renamed {image_counter-1} images successfully!")
