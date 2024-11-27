import os
import shutil
import zipfile
import sys
from torch.utils.data import Dataset
from PIL import Image

class DatasetManager:
    """Handles dataset download, extraction and preparation."""
    def __init__(self, zip_path, target_dir):
        """
        Initialize the DatasetManager.
        
        Args:
            zip_path (str): Path to the dataset zip file
            target_dir (str): Directory where the dataset should be extracted
        """
        self.zip_path = os.path.abspath(zip_path)
        self.target_dir = os.path.abspath(target_dir)

    def setup_dataset(self):
        """
        Extract and organize the dataset.
        Creates the target directory structure and extracts the dataset.
        """
        print("\nSetting up dataset...")
        print(f"Zip file: {self.zip_path}")
        print(f"Target directory: {self.target_dir}")
        
        # Create target directory if it doesn't exist
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Check if zip file exists
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Dataset zip file not found at: {self.zip_path}")
        
        # Check if it's actually a zip file
        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"File is not a valid zip file: {self.zip_path}")
            
        try:
            # Extract the archive using zipfile instead of shutil
            print(f"Extracting dataset from {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get the total size for progress reporting
                total_size = sum(file.file_size for file in zip_ref.filelist)
                extracted_size = 0
                
                # Extract each file with progress reporting
                for file in zip_ref.filelist:
                    zip_ref.extract(file, self.target_dir)
                    extracted_size += file.file_size
                    progress = (extracted_size / total_size) * 100
                    sys.stdout.write(f"\rExtraction progress: {progress:.1f}%")
                    sys.stdout.flush()
                
                print("\nExtraction completed!")
            
            # Rename the extracted folder to a standardized name
            old_path = os.path.join(self.target_dir, "New Plant Diseases Dataset(Augmented)")
            new_path = os.path.join(self.target_dir, "plantDataset")
            
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    print(f"Removing existing dataset at {new_path}")
                    shutil.rmtree(new_path)
                print(f"Renaming dataset folder to {new_path}")
                os.rename(old_path, new_path)
                print(f"Dataset extracted and organized at: {new_path}")
            else:
                # List contents of target directory for debugging
                print(f"\nWarning: Expected folder not found at: {old_path}")
                print("Contents of target directory:")
                for item in os.listdir(self.target_dir):
                    print(f"- {item}")
                raise FileNotFoundError(f"Expected dataset folder not found after extraction")
                
            # Verify the dataset structure
            self._verify_dataset_structure(new_path)
                
        except zipfile.BadZipFile as e:
            print(f"Error: Corrupt zip file - {str(e)}")
            raise
        except PermissionError as e:
            print(f"Error: Permission denied - {str(e)}")
            raise
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            print("Current directory structure:")
            self._print_directory_structure(self.target_dir)
            raise

    def _verify_dataset_structure(self, dataset_path):
        """
        Verify that the dataset has the expected directory structure.
        
        Args:
            dataset_path (str): Path to the extracted dataset
        """
        required_dirs = ['train', 'valid']
        
        print("\nVerifying dataset structure...")
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path):
                print(f"Error: Required directory not found: {dir_path}")
                print("Current directory structure:")
                self._print_directory_structure(dataset_path)
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
            
            # Check if the directory contains class folders
            class_dirs = [d for d in os.listdir(dir_path) 
                         if os.path.isdir(os.path.join(dir_path, d))]
            
            if not class_dirs:
                raise ValueError(f"No class directories found in {dir_path}")
            
            print(f"Found {len(class_dirs)} classes in {dir_name} directory")

    def _print_directory_structure(self, startpath):
        """
        Print the directory structure for debugging purposes.
        
        Args:
            startpath (str): The root directory to start printing from
        """
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files[:5]:  # Show only first 5 files to avoid clutter
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... ({len(files)-5} more files)")

    def get_class_names(self):
        """
        Get the list of class names from the training directory.
        
        Returns:
            list: List of class names
        """
        train_dir = os.path.join(self.target_dir, "plantDataset", "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found at: {train_dir}")
            
        class_names = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        return sorted(class_names)

class CustomImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.image_paths = []
        self.labels = []
        
        # Build list of image paths and labels
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_name))
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        return image, label
