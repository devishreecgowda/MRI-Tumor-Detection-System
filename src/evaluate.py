import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from preprocessing import preprocess_image


TEST_DIR = "../dataset/Testing"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def load_test_data(test_dir):
    images = []
    labels = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)

            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(preprocess_image(image_path))
                labels.append(CLASS_NAMES.index(class_name))

    return np.array(images), np.array(labels)


def evaluate_model():
    model = load_model("../models/mri_tumor_model.h5")

    test_images, true_labels = load_test_data(TEST_DIR)

    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(true_labels, predicted_labels,
                              target_names=CLASS_NAMES))

    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap="Blues")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("MRI Tumor Classification - Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
