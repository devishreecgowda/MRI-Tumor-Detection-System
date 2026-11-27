import cv2
import numpy as np

def preprocess_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess an MRI image for model input.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.resize(image, image_size)
    image = image / 255.0

    return image
