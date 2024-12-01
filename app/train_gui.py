import shutil
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from app.config import Config
from app.predict import DEFAULT_CLASS_NAMES
from app.models.model import ModelBuilder
from app.data.dataset import CustomImageDataset
from app.data.transforms import DataTransforms
from app.utils.visualization import plot_model_performance
from app.train import main as train_main
from utils.dataset_manager import DatasetManager



class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Model Training")

        # Make the window resizable
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Main container with padding
        main_frame = ttk.Frame(root, padding="40")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)

        # Initialize variables
        self.action_var = tk.StringVar(value="train")
        self.epochs_var = tk.StringVar()
        self.lr_var = tk.StringVar()
        self.batch_size_var = tk.StringVar()
        self.existing_model_var = tk.StringVar()
        self.model_arch_var = tk.StringVar(value="resnet50")  # Default to ResNet-50

        # Title Section
        title_label = ttk.Label(
            main_frame,
            text="Plant Disease Prediction",
            font=("Helvetica", 36, "bold"),
            foreground="#19783B",
        )
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Train or evaluate plant disease detection models",
            font=("Helvetica", 12),
            foreground="#666666",
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 20))

        # About Button
        about_button = ttk.Button(
            main_frame,
            text="About This Project",
            style="Link.TButton",
            command=self.show_about,
        )
        about_button.grid(row=2, column=0, pady=(0, 30))

        # Model Configuration Section
        model_frame = ttk.LabelFrame(
            main_frame, text="Model Configuration", padding="20"
        )
        model_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        model_frame.columnconfigure(0, weight=1)

        # Action Selection
        ttk.Label(model_frame, text="Select Action:", font=("Helvetica", 10)).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10)
        )

        action_frame = ttk.Frame(model_frame)
        action_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Radiobutton(
            action_frame,
            text="Train New Model",
            variable=self.action_var,
            value="train",
            command=self.toggle_options,
        ).grid(row=0, column=0, padx=(0, 20))

        ttk.Radiobutton(
            action_frame,
            text="Evaluate Existing Model",
            variable=self.action_var,
            value="evaluate",
            command=self.toggle_options,
        ).grid(row=0, column=1)

        # Training Frame
        self.training_frame = ttk.Frame(model_frame)
        self.training_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        self.training_frame.columnconfigure(1, weight=1)

        # Training Parameters
        params = [
            ("Epochs:", self.epochs_var, "10"),
            ("Learning Rate:", self.lr_var, "0.001"),
            ("Batch Size:", self.batch_size_var, "32"),
        ]

        # Add model architecture selector before training parameters
        ttk.Label(self.training_frame, text="Model Architecture:", font=("Helvetica", 10)).grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        model_arch_combo = ttk.Combobox(
            self.training_frame,
            textvariable=self.model_arch_var,
            values=["resnet50", "efficientnet"],
            state="readonly",
            width=15
        )
        model_arch_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Adjust row numbers for other parameters
        for idx, (label, var, default) in enumerate(params):
            ttk.Label(self.training_frame, text=label, font=("Helvetica", 10)).grid(
                row=idx+1, column=0, sticky=tk.W, pady=5  # Add 1 to row index
            )
            entry = ttk.Entry(self.training_frame, textvariable=var, width=15)
            entry.grid(row=idx+1, column=1, sticky=tk.W, padx=(10, 0))  # Add 1 to row index
            var.set(default)

        # Evaluation Frame
        self.eval_frame = ttk.Frame(model_frame)
        self.eval_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        self.eval_frame.columnconfigure(0, weight=1)

        ttk.Label(
            self.eval_frame, text="Select Model to Evaluate:", font=("Helvetica", 10)
        ).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_list = ttk.Combobox(
            self.eval_frame, textvariable=self.existing_model_var, width=30
        )
        self.model_list.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Initially hide evaluation frame
        self.eval_frame.grid_remove()

        # Action Button (using tk.Button for custom colors)
        self.action_button = tk.Button(
            main_frame,
            text="Start Training",
            command=self.execute_action,
            bg="#19783B",  # Green background
            fg="white",  # White text
            font=("Helvetica", 10),
            relief="flat",
            padx=20,
            pady=10,
        )
        self.action_button.grid(row=4, column=0, pady=30)

        # Progress Bar
        self.progress = ttk.Progressbar(
            main_frame,
            length=400,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar",
        )
        self.progress.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            foreground="#666666",
        ).grid(row=6, column=0)

        # Add a text widget for logging
        self.log_text = tk.Text(main_frame, height=10, width=50)
        self.log_text.grid(row=7, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Add scrollbar for log
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=7, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Make the log read-only
        self.log_text.configure(state='disabled')

        # Add dataset setup frame after model configuration section
        dataset_frame = ttk.LabelFrame(
            main_frame, text="Dataset Configuration", padding="20"
        )
        dataset_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        dataset_frame.columnconfigure(0, weight=1)
        
        # Dataset setup button
        self.setup_dataset_button = ttk.Button(
            dataset_frame,
            text="Setup Datasets",
            command=self.setup_datasets
        )
        self.setup_dataset_button.grid(row=0, column=0, pady=5)
        
        # Dataset status
        self.dataset_status_var = tk.StringVar(value="Datasets not verified")
        ttk.Label(
            dataset_frame,
            textvariable=self.dataset_status_var,
            font=("Helvetica", 10),
            foreground="#666666"
        ).grid(row=1, column=0, pady=5)
        
        # Add warning note
        warning_text = "Note: Only use this setup when:\n1. First time initializing the dataset\n2. Need to clean and refresh existing dataset"
        warning_label = ttk.Label(
            dataset_frame,
            text=warning_text,
            font=("Helvetica", 9),
            foreground="#CC0000",  # Red color for warning
            justify=tk.LEFT,
            wraplength=400
        )
        warning_label.grid(row=2, column=0, pady=(5, 10), sticky='w')
        
        # Adjust the grid row for existing elements
        self.action_button.grid(row=5, column=0, pady=30)  # Change from row=4
        self.progress.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))  # Change from row=5
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            foreground="#666666",
        ).grid(row=7, column=0)  # Change from row=6

        # Configure styles
        self.configure_styles()

    def configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()

        # Configure link button style
        style.configure("Link.TButton", font=("Helvetica", 10), foreground="#0066cc")

        # Configure progress bar style
        style.configure(
            "Modern.Horizontal.TProgressbar",
            background="#19783B",
            troughcolor="#f0f0f0",
        )

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Plant Disease Prediction\n\n"
            "This application helps train and evaluate models "
            "for detecting diseases in plant leaves using deep learning.",
        )

    def update_model_list(self):
        """Update the list of available models"""
        model_dir = os.path.join(Config.BASE_DIR, "models")
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            self.model_list["values"] = models

    def toggle_options(self):
        """Toggle visibility of options based on selected action"""
        if self.action_var.get() == "train":
            self.training_frame.grid()
            self.eval_frame.grid_remove()
            self.action_button.configure(text="Start Training")
        else:
            self.training_frame.grid_remove()
            self.eval_frame.grid()
            self.action_button.configure(text="Start Evaluation")
            self.update_model_list()

    def execute_action(self):
        """Execute the selected action (training or evaluation)"""
        try:
            if self.action_var.get() == "train":
                self.train_model()
            else:
                self.evaluate_model()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_log(self, message):
        """Update the log text widget with a new message"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)  # Scroll to bottom
        self.log_text.configure(state='disabled')
        self.root.update()

    def train_model(self):
        """Start the training process"""
        try:
            # Clear previous log
            self.log_text.configure(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.configure(state='disabled')
            
            # Get training parameters
            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            batch_size = int(self.batch_size_var.get())
            model_arch = self.model_arch_var.get()
            
            # Validate parameters
            if epochs <= 0 or lr <= 0 or batch_size <= 0:
                messagebox.showerror("Error", "Please enter valid positive numbers for all parameters")
                return
            
            self.status_var.set("Training model...")
            self.progress['value'] = 0
            self.root.update()
            
            # Create a custom progress callback
            def progress_callback(epoch, total_epochs, batch=None, total_batches=None, 
                                loss=None, val_loss=None, val_acc=None):
                # Update progress bar
                progress = (epoch / total_epochs) * 100
                self.progress['value'] = progress
                
                # Update status
                if batch is not None:
                    status = f"Epoch {epoch}/{total_epochs}, Batch {batch}/{total_batches}"
                    if loss is not None:
                        status += f", Loss: {loss:.4f}"
                    if val_loss is not None:
                        status += f", Val Loss: {val_loss:.4f}"
                    if val_acc is not None:
                        status += f", Val Acc: {val_acc:.2f}%"
                    
                    self.status_var.set(status)
                    self.update_log(status)
                
                self.root.update()
            
            # Start training with progress callback
            train_main(
                epochs=epochs, 
                lr=lr, 
                batch_size=batch_size, 
                model_type=model_arch,
                progress_callback=progress_callback
            )
            
            self.status_var.set("Training completed")
            self.progress['value'] = 100
            self.update_log("Training completed successfully!")
            
            messagebox.showinfo("Success", "Training completed successfully!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters")
        except Exception as e:
            self.status_var.set("Error during training")
            error_msg = str(e)
            self.update_log(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def evaluate_model(self):
        try:
            self.status_var.set("Loading model...")
            self.progress['value'] = 0
            self.root.update()

            # Get the number of classes from DEFAULT_CLASS_NAMES
            num_classes = len(DEFAULT_CLASS_NAMES)
            device = Config.DEVICE
            
            # Create model with correct parameters
            model = ModelBuilder.create_model(
                num_classes=num_classes,
                model_type=self.model_arch_var.get(),
                pretrained=False
            ).to(device)
            
            # Load checkpoint
            model_path = os.path.join(Config.MODEL_DIR, self.existing_model_var.get())
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load state dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Prepare test dataset
            test_dataset = CustomImageDataset(
                root_dir=os.path.join(Config.BASE_DIR, "data", "plantDataset"),
                split='test',
                transform=DataTransforms().get_test_transform()
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=32, 
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            # Initialize lists for predictions and labels
            all_preds = []
            all_labels = []
            
            # Evaluate model
            self.status_var.set("Evaluating model...")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Move to CPU for numpy conversion
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress
                    progress = (batch_idx + 1) / len(test_loader) * 100
                    self.progress['value'] = progress
                    self.root.update()
            
            # Calculate confusion matrix
            conf_mat = confusion_matrix(all_labels, all_preds)
            
            # Plot and save results
            self.status_var.set("Generating performance plots...")
            plot_model_performance(
                confusion_mat=conf_mat,
                class_names=DEFAULT_CLASS_NAMES,
                save_path=os.path.join(Config.BASE_DIR, "static", "plots", "model_performance.png")
            )
            
            # Calculate and display metrics
            class_names = [f"{plant}-{condition}" for plant, condition in DEFAULT_CLASS_NAMES.values()]
            report = classification_report(all_labels, all_preds, target_names=class_names)
            
            # Show results
            self._show_evaluation_results(report)
            
            self.status_var.set("Evaluation completed")
            self.progress['value'] = 100
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            raise e

    def _show_evaluation_results(self, metrics_text):
        """Show evaluation results in a new window"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Model Evaluation Results")
        results_window.geometry("800x600")
        
        # Add scrolled text widget
        text_widget = scrolledtext.ScrolledText(results_window, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Insert metrics text
        text_widget.insert(tk.END, metrics_text)
        text_widget.config(state='disabled')

    def setup_datasets(self):
        """Setup both plant and landscape datasets"""
        try:
            self.status_var.set("Setting up datasets...")
            self.progress['value'] = 0
            self.root.update()

            # Ask user for plant dataset zip location
            plant_zip = filedialog.askopenfilename(
                title="Select Plant Disease Dataset ZIP",
                filetypes=[("ZIP files", "*.zip")]
            )
            
            # Ask user for landscape dataset zip location
            landscape_zip = filedialog.askopenfilename(
                title="Select Landscape Dataset ZIP",
                filetypes=[("ZIP files", "*.zip")]
            )
            
            if not plant_zip or not landscape_zip:
                messagebox.showerror("Error", "Both dataset ZIP files must be selected")
                return

            # Setup plant dataset
            self.status_var.set("Extracting plant dataset...")
            self.progress['value'] = 25
            self.root.update()
            
            plant_manager = DatasetManager(plant_zip, os.path.join(Config.BASE_DIR, "data"))
            plant_manager.setup_dataset()
            
            # Setup landscape dataset
            self.status_var.set("Extracting landscape dataset...")
            self.progress['value'] = 50
            self.root.update()
            
            landscape_manager = DatasetManager(landscape_zip, os.path.join(Config.BASE_DIR, "data"))
            landscape_manager.setup_landscape_dataset()  # New method needed in DatasetManager
            
            # Merge datasets
            self.status_var.set("Merging datasets...")
            self.progress['value'] = 75
            self.root.update()
            
            self._merge_datasets()
            
            self.status_var.set("Datasets setup completed!")
            self.dataset_status_var.set("Datasets verified ✓")
            self.progress['value'] = 100
            messagebox.showinfo("Success", "Datasets have been set up successfully!")
            
        except Exception as e:
            self.status_var.set("Error setting up datasets")
            self.dataset_status_var.set("Dataset setup failed!")
            messagebox.showerror("Error", str(e))
            print(f"Error details: {str(e)}")

    def _merge_datasets(self):
        """Merge landscape dataset into plant dataset structure"""
        base_dir = os.path.join(Config.BASE_DIR, "data", "plantDataset")
        landscape_dir = os.path.join(Config.BASE_DIR, "data", "landscapeDataset")
        
        # Get the landscape class index from DEFAULT_CLASS_NAMES
        landscape_class_id = None
        for class_id, (plant, condition) in DEFAULT_CLASS_NAMES.items():
            if plant == "Landscape":
                landscape_class_id = class_id
                break
        
        if landscape_class_id is None:
            raise ValueError("Landscape class not found in DEFAULT_CLASS_NAMES")
        
        print(f"Moving landscape images to class index: {landscape_class_id}")
        
        # For each split (train/valid/test)
        for split in ['train', 'valid', 'test']:
            # Create landscape class directory using the class ID
            target_dir = os.path.join(base_dir, split, f"c{landscape_class_id}_landscape")
            source_dir = os.path.join(landscape_dir, split)
            
            # Create landscape class directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Move all landscape images to the target directory
            if os.path.exists(source_dir):
                print(f"Processing {split} split...")
                count = 0
                for root, _, files in os.walk(source_dir):
                    for img in files:
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_path = os.path.join(root, img)
                            # Add prefix to ensure unique filenames
                            dst_filename = f"landscape_{count}_{img}"
                            dst_path = os.path.join(target_dir, dst_filename)
                            shutil.copy2(src_path, dst_path)
                            count += 1
                print(f"Moved {count} images to {target_dir}")


def launch_gui():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
