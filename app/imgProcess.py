import os
import cv2
import numpy as np


def preprocess_images(image_dir, target_size=(224, 224)):
    images = []
    labels = []
    for label, plant_class in enumerate(os.listdir(image_dir)):
        class_folder = os.path.join(image_dir, plant_class)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, target_size)
                image = image / 255.0  # Normalize image to [0, 1]
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)


images, labels = preprocess_images("data")
