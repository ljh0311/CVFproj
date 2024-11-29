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
  + Future work includes:
    - Developing a custom CNN architecture specifically optimized for Singapore's native plant species
    - Ensemble methods combining predictions from both models
    - Additional model architectures based on performance analysis

* **Expected Output**:
  + The model will output a **disease label** for each input image, indicating whether the plant is healthy or which disease it is suffering from.
  + It will also provide a **confidence score**, representing the model’s certainty in its predictions.

## Deployment

The model will be deployed as a **web application** to provide an easy-to-use interface for users.

* **Web App Functionality**:
  + Users can upload images of plant leaves through the web interface.
  + Upon uploading, the web app will process the image and return the predicted disease label and confidence score.
  
* **Technologies Used**:
  + Flask for creating the web application.
  + PyTorch for building and deploying the deep learning model.
  + OpenCV and PIL for image processing.

## Flow Diagram

The workflow of the application can be summarized as follows:

1. **User uploads image** of a plant leaf through the web interface.
2. **Web server processes the image**, resizing and preparing it for prediction.
3. **Model predicts the class** (diseased or healthy) and the specific disease (if applicable).
4. **Results are displayed** to the user, including the disease label and the confidence score.

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

4. **Download the datasets**:

```bash
# Method 1: Using kaggle CLI (recommended)
kaggle datasets download vipoooool/new-plant-diseases-dataset
kaggle datasets download arnaud58/landscape-pictures

# Method 2: Using curl
curl -L -o leafachive.zip https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset
curl -L -o landscapePhotos.zip https://www.kaggle.com/api/v1/datasets/download/arnaud58/landscape-pictures

```

5. **Extract the datasets**:
* Extract both downloaded datasets into the `data` directory in the project root
* Ensure the directory structure matches:
  

```
  CVFProj/
  ├── data/
  │   ├── New Plant Diseases Dataset/
  │   └── PlantVillage/
  ```

6. **Train the model** (Required step):
```bash
python train.py
```
This step will:
- Create necessary model files (.pth) in the `models` directory
- Generate checkpoints in the `checkpoints` directory
- Train the model on your dataset
- Save class names and model configurations

7. **Run the Flask Web App**:

```bash
python app.py
```

8. Open the web interface at http://127.0.0.1:5000 to upload images and view predictions.

**Note**: 
- Make sure you have Python 3.8+ installed on your system before starting
- The training step (6) is required as model files are not included in the repository due to size limitations
- Training time will vary depending on your hardware configuration
