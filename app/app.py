from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Define available models and their file paths
models = {
    "cnn": "models/cnn_model.h5",
    "fine_tuned": "models/fine_tuned_model.h5",
}

# Load the models into memory
import os  # Import for file existence check
import tensorflow as tf

# Define available models and their file paths
models = {
    "cnn": "models/cnn_model.h5",
    "fine_tuned": "models/fine_tuned_model.h5",
}

# Load models into memory if the files exist
loaded_models = {}
for name, path in models.items():
    if os.path.exists(path):
        try:
            loaded_models[name] = tf.keras.models.load_model(path)
            print(f"Model '{name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{name}': {e}")
    else:
        print(f"Model file for '{name}' does not exist at {path}.")

# Define class labels
labels = ["Disease A", "Disease B", "Healthy"]  # Replace with your actual class names


@app.route("/")
def home():
    return render_template("index.html", model_names=models.keys())


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Get the selected model from the form
    selected_model_name = request.form.get("model")
    if not selected_model_name or selected_model_name not in models:
        return "Invalid model selected"

    model = loaded_models[selected_model_name]

    # Read and preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_name = labels[class_idx]

    return f"Predicted disease: {class_name} using {selected_model_name} model"


if __name__ == "__main__":
    app.run(debug=True)
