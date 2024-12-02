# Plantastic: Plant Disease Classification

**Plantastic** is a deep learning-based project that identifies plant diseases by analyzing leaf images and comparing them to patterns it has learned from thousands of other plant disease examples. The project uses computer vision and deep learning techniques to recognize visual symptoms and characteristics that indicate whether a plant is healthy or diseased, and can determine the specific type of disease based on these learned visual patterns.

## Version Control

This repository maintains two main branches:

1. **main**: Production-ready code
   * Contains only fully functional and tested code
   * Each push is versioned as "Version X.X"
   * All features must be thoroughly tested before merging

2. **fileFix**: Stable Backup branch
   * Contains only tested and working code changes
   * Serves as a stable backup point for the codebase
   * Updated only after changes are verified to be functional

**Note**: Contributors should only push to main when code is fully functional and tested.

## Data Source

The project uses two datasets:

1. **New Plant Diseases Dataset** by Samir Bhattarai
   * Available on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   * Contains **87, 000** colored images of healthy and diseased crop leaves
   * **38 classes** of different plants in both healthy and diseased states
   * Download: `https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset`

2. **Landscape Pictures Dataset**
   * Available on [Kaggle](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
   * Contains diverse environmental and landscape images
   * Useful for testing model robustness with non-plant images
   * Download: `https://www.kaggle.com/api/v1/datasets/download/arnaud58/landscape-pictures`

## Model

The core of this project is a **classification model** used to predict the health status of plant leaves and identify any diseases.

* **Model Approach**:
  + Currently supporting two models for transfer learning and fine-tuning:
    - **ResNet-50**: A deep residual network known for its ability to train very deep networks effectively
    - **EfficientNet-B0**: A lightweight model that balances model size and accuracy through compound scaling
  + Models can be selected through the web interface for different use cases:
    - ResNet-50 for higher accuracy on complex disease patterns
    - EfficientNet-B0 for faster inference and mobile deployment
    - Configurable training parameters:
      * Number of epochs for training iterations
      * Learning rate for optimization control 
      * Batch size for memory and performance tuning
  + Future work includes:
    - Developing a custom CNN architecture specifically optimized for Singapore's native plant species
    - Ensemble methods combining predictions from both models
    - Additional model architectures based on performance analysis

* **Expected Output**:
  + The model will output a **disease label** for each input image, indicating whether the plant is healthy or which disease it is suffering from.
  + It will also provide a **confidence score**, representing the model’s certainty in its predictions.

## Deployment

The project offers two interfaces for users:

### 1. Desktop GUI Application

A Tkinter-based GUI application that provides comprehensive model training and dataset management capabilities:

* **Dataset Management**:
  + Setup and organize both plant disease and landscape datasets
  + Extract and merge datasets automatically
  + Visual progress indicators for dataset operations
  + Dataset verification and status updates

* **Model Configuration**:
  + Choose between training new models or evaluating existing ones
  + Select model architecture (ResNet-50 or EfficientNet-B0)
  + Configure training parameters (epochs, learning rate, batch size)
  + Real-time training progress visualization
  + Training logs and status updates

* **Model Evaluation**:
  + Evaluate trained models on test datasets
  + View confusion matrices and classification reports
  + Performance metrics visualization

### 2. Web Interface

A Flask-based web application for easy disease prediction:

* **Prediction Interface**:
  + Upload plant leaf images for disease detection
  + Select between available trained models
  + View prediction results with confidence scores
  + Support for multiple image formats (JPG, JPEG, PNG)

* **Data Contribution**:
  + Contribute to the training dataset
  + Upload multiple images simultaneously (up to 10)
  + Select appropriate disease labels
  + Guidelines for proper image submission

* **User Features**:
  + View available disease classes
  + Access project information and documentation
  + Dynamic background image rotation
  + Responsive design for various screen sizes

## Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/ljh0311/CVFproj
cd CVFProj
```

2. **Create and activate a virtual environment** (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download and setup datasets**:
   * Rename the downloaded zip files:
     - Plant disease dataset → `leafarchive.zip`
     - Landscape dataset → `landscapePhotos.zip`
   * Place both zip files in the `data` directory
   
   * Use the GUI application for dataset setup:
     ```bash
     python -m app.train_gui
     ```
   * For automatic setup:
     - Click "Setup Datasets" and select the downloaded ZIP files when prompted
     - The GUI will handle extraction and verification automatically
   
   * For manual setup:
     - Extract plantDataset.zip which should contain:
       + test/
       + train/ 
       + valid/
     - Extract landscapeDataset.zip directly into a folder
       + Should contain only image files (.jpg, .jpeg, .png)
       + No subfolders needed
     - Verify folder structure is correct before proceeding

5. **Train the model** (Required step):
   * Using the GUI:
     - Select "Train New Model"
     - Configure training parameters
     - Click "Start Training"
   * Or using command line:
     ```bash
     python train.py
     ```

6. **Launch the application**:
   * For GUI:
     ```bash
     python -m app.train_gui
     ```
   * For web interface:
     ```bash
     python app.py
     ```
   * Access web interface at http://127.0.0.1:5000

**Note**: 
- Python 3.8+ required
- Training step is mandatory as model files are not included
- GPU recommended for faster training
- Web interface requires trained models in the `models` directory
