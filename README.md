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
  * A **Convolutional Neural Network (CNN)** will be trained from scratch to perform image classification.
  * Alternatively, we may choose to **fine-tune a pre-trained model** (e.g., VGG16, ResNet) to improve model accuracy and reduce training time.

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
  * TensorFlow/Keras for building and deploying the deep learning model.
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

2. **Install dependencies**:

```bash
pip install -r requirements.txt

3. Run the Flask Web App

```bash
   python app.py
   ```

4. Open the web interface at <http://127.0.0.1:5000> to upload images and view predictions.
