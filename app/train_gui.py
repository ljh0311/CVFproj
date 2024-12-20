import shutil
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import scrolledtext
from app.config import Config  # Changed from app.config
from app.train import main as train_main, get_next_version  # Changed from app.train
from app.predict import DEFAULT_CLASS_NAMES
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
        self.root.minsize(600, 800)  # Add minimum window size

        # Create a canvas with scrollbar for the entire content
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create main layout frames
        left_frame = ttk.Frame(scrollable_frame, padding="20")
        right_frame = ttk.Frame(scrollable_frame, padding="20")
        
        # Configure grid weights for the frames
        scrollable_frame.columnconfigure(0, weight=3)  # Left frame takes 3/4
        scrollable_frame.columnconfigure(1, weight=1)  # Right frame takes 1/4
        
        # Place frames in grid
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Move all existing content to left_frame instead of main_frame
        # (Replace main_frame with left_frame in all the widget definitions)

        # Log Text - Move to right frame
        log_frame = ttk.LabelFrame(right_frame, text="Training Log", padding="10")
        log_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=30, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Add scrollbar for log
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add dataset setup frame after model configuration section
        dataset_frame = ttk.LabelFrame(
            left_frame, text="Dataset Configuration", padding="20"
        )
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
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
        
        # Configure styles
        self.configure_styles()

        # Initialize variables
        self.action_var = tk.StringVar(value="train")
        self.epochs_var = tk.StringVar()
        self.lr_var = tk.StringVar()
        self.batch_size_var = tk.StringVar()
        self.existing_model_var = tk.StringVar()
        self.model_arch_var = tk.StringVar(value="resnet50")

        # Title Section in left_frame
        title_label = ttk.Label(
            left_frame,
            text="Plant Disease Prediction",
            font=("Helvetica", 36, "bold"),
            foreground="#19783B",
        )
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        subtitle_label = ttk.Label(
            left_frame,
            text="Train or evaluate plant disease detection models",
            font=("Helvetica", 12),
            foreground="#666666",
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 20))

        about_button = ttk.Button(
            left_frame,
            text="About This Project",
            style="Link.TButton",
            command=self.show_about,
        )
        about_button.grid(row=2, column=0, pady=(0, 30))

        # Model Configuration Section
        model_frame = ttk.LabelFrame(
            left_frame, text="Model Configuration", padding="20"
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

        # Model Architecture selector
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

        # Training Parameters
        params = [
            ("Epochs:", self.epochs_var, "10"),
            ("Learning Rate:", self.lr_var, "0.001"),
            ("Batch Size:", self.batch_size_var, "32"),
        ]

        for idx, (label, var, default) in enumerate(params, start=1):
            ttk.Label(self.training_frame, text=label, font=("Helvetica", 10)).grid(
                row=idx, column=0, sticky=tk.W, pady=5
            )
            entry = ttk.Entry(self.training_frame, textvariable=var, width=15)
            entry.grid(row=idx, column=1, sticky=tk.W, padx=(10, 0))
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

        # Progress section
        progress_frame = ttk.Frame(left_frame)
        progress_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)

        # Action Button
        self.action_button = tk.Button(
            progress_frame,
            text="Start Training",
            command=self.execute_action,
            bg="#19783B",
            fg="white",
            font=("Helvetica", 10),
            relief="flat",
            padx=20,
            pady=10,
        )
        self.action_button.grid(row=0, column=0, pady=(0, 10))

        # Progress Bar
        self.progress = ttk.Progressbar(
            progress_frame,
            length=400,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            foreground="#666666",
        ).grid(row=2, column=0, pady=(0, 5))

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
            
            # Generate model name with parameters
            model_name = f"{model_arch}_{epochs}_{lr}_{batch_size}"
            
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
            
            # Start training with progress callback and model name
            train_main(
                epochs=epochs, 
                lr=lr, 
                batch_size=batch_size, 
                model_type=model_arch,
                model_name=model_name,
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

            device = Config.DEVICE
            
            # Initialize model with correct number of classes
            model = ModelBuilder.create_model(
                num_classes=len(DEFAULT_CLASS_NAMES),
                model_type=self.model_arch_var.get()
            ).to(device)
            
            # Load checkpoint with weights_only=True and handle different model file formats
            model_path = os.path.join(Config.MODEL_DIR, self.existing_model_var.get())
            try:
                # Try loading as a state dict first
                checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # If it's just the state dict directly
                    model.load_state_dict(checkpoint)
            except Exception as e:
                self.update_log(f"Error loading model: {str(e)}")
                raise
            
            # Set model to evaluation mode
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

            # Setup plant dataset
            plant_manager = DatasetManager(
                Config.PLANT_ZIP_PATH,
                os.path.join(Config.BASE_DIR, "data")
            )
            plant_manager.setup_dataset(dataset_type="plant")
            
            # Setup landscape dataset
            landscape_manager = DatasetManager(
                Config.LANDSCAPE_ZIP_PATH,
                os.path.join(Config.BASE_DIR, "data")
            )
            landscape_manager.setup_dataset(dataset_type="landscape")
            
            self.status_var.set("Datasets setup completed!")
            self.dataset_status_var.set("Datasets verified ✓")
            self.progress['value'] = 100
            messagebox.showinfo("Success", "Datasets have been set up successfully!")
            
        except Exception as e:
            self.status_var.set("Error setting up datasets")
            self.dataset_status_var.set("Dataset setup failed!")
            messagebox.showerror("Error", str(e))
            print(f"Error details: {str(e)}")


def launch_gui():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
