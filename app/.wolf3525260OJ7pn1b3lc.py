import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imgProcess import preprocess_images
from model import create_cnn_model, create_fine_tuned_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm.notebook import tqdm


# Define parameters
ROOT_DIR = "C:/Users/user/Documents/SITstuffs/CompVis/CVFproj"
DATA_DIR = os.path.join(ROOT_DIR, "data/plantDataset")
print('DATA_DIR: ', DATA_DIR)

IMAGE_SIZE = (224, 224)  # Expected image size for the model
EPOCHS = 10
BATCH_SIZE = 32
MODEL_TO_TRAIN = "fine_tuned"  # Can be 'cnn' or 'fine_tuned'


# Load images and labels using the preprocessing function
images, labels = preprocess_images(DATA_DIR, target_size=IMAGE_SIZE)


# Helper function to format the label names
def format_label(label):
    label = label.split(".")[-1]  # Assume label is the file name's suffix
    label = label.replace("_", " ")  # Replace underscores with spaces
    label = label.title()  # Capitalize the first letter of each word
    return label.replace(" ", "")  # Remove spaces after title casing


# Extract class labels from the dataset
def get_classes(data_dir):
    return [
        format_label(c)
        for c in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, c))
    ]


# Retrieve and print classes
classes = get_classes(DATA_DIR)
print(f"Classes: {classes}")


# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
print(f"Training data shape: {X_train.shape}, Training labels: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels: {y_val.shape}")


# Select the model to train based on the model_to_train variable
def model_chosen(model_to_train):
    if model_to_train == "cnn":
        return create_cnn_model()
    elif model_to_train == "fine_tuned":
        return create_fine_tuned_model()
    else:
        raise ValueError("Invalid model type specified. Choose 'cnn' or 'fine_tuned'.")


# Set up callbacks for model training
checkpoint = ModelCheckpoint(
    "models/best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
)

early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

# Instantiate the chosen model
model = model_chosen(MODEL_TO_TRAIN)

# Train the model with callbacks
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping],
)

N = 5

# Randomly select N samples from the batch
random_indices = random.sample(range(BATCH_SIZE), N)
random_images = images[random_indices]
random_labels = labels[random_indices]

# Plot the random images
fig, axes = plt.subplots(1, N, figsize=(N * 2, 2))
for i in range(N):
    img = random_images[i].permute(
        1, 2, 0
    )  # Reorder the tensor for plotting (channels last)
    # Normalize the image data to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    axes[i].imshow(img.numpy())  # Convert tensor to numpy for display
    axes[i].set_title(f"{classes[random_labels[i].item()]}")
    axes[i].axis("off")  # Hide axes for cleaner visualization

plt.tight_layout()
plt.show()

# Evaluate the model on the validation set
model.evaluate(X_val, y_val)

# Save the final model after training completes
model.save(f"models/{MODEL_TO_TRAIN}_final_model.h5")
