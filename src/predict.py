import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import preprocess_image


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "mri_tumor_model.h5"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def predict_image(image_path):
    model = load_model(MODEL_PATH)

    image = preprocess_image(image_path, augment=False)
    image = np.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)

    predicted_index = np.argmax(probabilities[0])
    confidence = float(probabilities[0][predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    if predicted_class == "notumor":
        result = "No Tumor Detected"
    else:
        result = f"Predicted Category: {predicted_class}"

    print(f"\nResult: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_mri_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
