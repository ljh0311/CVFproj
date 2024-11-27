# Plantastic: Plant Disease Classification

**Plantastic** is a deep learning-based project that aims to identify plant diseases by analyzing images of plant leaves. The project uses computer vision and deep learning techniques to classify images into healthy or diseased categories and further classify the disease type.

## Data Source

The dataset used in this project is the **New Plant Diseases Dataset** by Samir Bhattarai, available on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

* **Dataset Overview**:
  * Contains **87, 000** colored images of healthy and diseased crop leaves.
  * The dataset is divided into **38 classes**, representing different plants in both healthy and diseased states.
  
## Model

The core of this project is a **classification model** used to predict the health status of plant leaves and identify any diseases.

* **Model Approach**:
  * Currently using **ResNet-50** as the base model for transfer learning and fine-tuning.
  * Future work includes developing a custom CNN architecture specifically optimized for Singapore's native plant species and their unique disease patterns.

* **Expected Output**:
  * The model will output a **disease label** for each input image, indicating whether the plant is healthy or which disease it is suffering from.
  * It will also provide a **confidence score**, representing the model’s certainty in its predictions.

## Deployment

The model will be deployed as a **web application** to provide an easy-to-use interface for users.

* **Web App Functionality**:
  * Users can upload images of plant leaves through the web interface.
  * Upon uploading, the web app will process the image and return the predicted disease label and confidence score.
  
* **Technologies Used**:
  * Flask for creating the web application.
  * PyTorch for building and deploying the deep learning model.
  * OpenCV and PIL for image processing.

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

4. **Download the dataset**:
- Download the New Plant Diseases Dataset from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Extract the downloaded dataset into the `data` directory in the project root
- Ensure the directory structure matches:
  ```
  CVFProj/
  ├── data/
  │   └── New Plant Diseases Dataset/
  ```

5. **Train the model** (optional, skip if using pre-trained weights):
```bash
python train.py
```

6. **Run the Flask Web App**:
```bash
python app.py
```

7. Open the web interface at http://127.0.0.1:5000 to upload images and view predictions.

**Note**: Make sure you have Python 3.8+ installed on your system before starting.
