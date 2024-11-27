import os
import shutil
import zipfile
import sys


class DatasetManager:
    """Handles dataset download, extraction and preparation."""

    def __init__(self, zip_path, target_dir):
        self.zip_path = os.path.abspath(zip_path)
        self.target_dir = os.path.abspath(target_dir)

    def setup_dataset(self):
        """Extract and organize the dataset."""
        print("\nSetting up dataset...")
        print(f"Target directory: {self.target_dir}")

        # Create target directory if it doesn't exist
        os.makedirs(self.target_dir, exist_ok=True)

        # Download dataset from Kaggle
        print("Downloading dataset from Kaggle...")
        download_path = os.path.expanduser("~/Downloads/archive.zip")
        try:
            os.system(
                f"curl -L -o {download_path} "
                "https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")

        if not os.path.exists(download_path):
            raise FileNotFoundError(f"Failed to download dataset to: {download_path}")

        if not zipfile.is_zipfile(download_path):
            raise ValueError(
                f"Downloaded file is not a valid zip file: {download_path}"
            )

        try:
            # Clean up existing directories
            dataset_path = os.path.join(self.target_dir, "plantDataset")
            temp_dir = os.path.join(self.target_dir, "temp_extract")

            for path in [dataset_path, temp_dir]:
                if os.path.exists(path):
                    print(f"Removing existing directory: {path}")
                    shutil.rmtree(path)

            # Create fresh directories
            os.makedirs(temp_dir)
            os.makedirs(dataset_path)

            # Extract the zip file
            print(f"Extracting dataset from {download_path}...")
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            print("Extraction completed!")

            # Find the source directories
            base_path = os.path.join(temp_dir, "New Plant Diseases Dataset(Augmented)")
            if os.path.exists(
                os.path.join(base_path, "New Plant Diseases Dataset(Augmented)")
            ):
                base_path = os.path.join(
                    base_path, "New Plant Diseases Dataset(Augmented)"
                )

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
            os.remove(download_path)

            # Verify the final structure
            self._verify_dataset_structure(dataset_path)
            print("Dataset setup completed successfully!")

        except Exception as e:
            print(f"\nError processing dataset: {str(e)}")
            # Clean up on error
            for path in [temp_dir, dataset_path]:
                if os.path.exists(path):
                    shutil.rmtree(path)
            if os.path.exists(download_path):
                os.remove(download_path)
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
