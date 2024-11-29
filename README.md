# Plantastic: Plant Disease Classification

**Plantastic** is a deep learning-based project that aims to identify plant diseases by analyzing images of plant leaves. The project uses computer vision and deep learning techniques to classify images into healthy or diseased categories and further classify the disease type.

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
   * Use the GUI application for dataset setup:
     ```bash
     python -m app.train_gui
     ```
   * Click "Setup Datasets" and select the downloaded ZIP files when prompted
   * The GUI will automatically:
     - Extract both datasets
     - Merge them into the correct structure
     - Verify the setup

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
