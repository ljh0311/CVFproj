import os
from sklearn.model_selection import train_test_split
from app.imgProcess import preprocess_images
from app.model import create_cnn_model, create_fine_tuned_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define parameters
data_dir = "CVFProj/data"
image_size = (224, 224)  # Expected image size for the model
epochs = 10
batch_size = 32
model_to_train = "fine_tuned"  # Can be 'cnn' or 'fine_tuned'

# Load images and labels using the preprocessing function
images, labels = preprocess_images(data_dir, target_size=image_size)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Select the model to train based on the model_to_train variable
if model_to_train == "cnn":
    model = create_cnn_model()
elif model_to_train == "fine_tuned":
    model = create_fine_tuned_model()
else:
    raise ValueError("Invalid model type specified. Choose 'cnn' or 'fine_tuned'.")

# Set up callbacks
checkpoint = ModelCheckpoint(
    "models/best_model.h5",  # Path where the best model will be saved
    monitor="val_loss",  # Monitor validation loss to save the best model
    save_best_only=True,
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss",  # Stop training if the validation loss stops improving
    patience=3,  # Number of epochs to wait before stopping
    verbose=1,
)

# Train the selected model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stopping],  # Use the callbacks for better training
)

# Evaluate the model on the validation set
model.evaluate(X_val, y_val)

# Optionally, save the final model (after training completes)
model.save(f"models/{model_to_train}_final_model.h5")
