import os
import numpy as np

from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from preprocessing import preprocess_image


# Dataset paths
TRAIN_DIR = "../dataset/Training"

IMAGE_SIZE = 128
BATCH_SIZE = 20
EPOCHS = 5

LABEL_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}


def load_data(directory):
    paths, labels = [], []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)

        if not os.path.isdir(label_path):
            continue

        for image in os.listdir(label_path):
            if image.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(label_path, image))
                labels.append(label)

    return shuffle(paths, labels, random_state=42)


def data_generator(paths, labels, batch_size=BATCH_SIZE):
    while True:
        indices = np.arange(len(paths))
        np.random.shuffle(indices)

        for i in range(0, len(paths), batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_paths = [paths[idx] for idx in batch_indices]
            batch_labels = [LABEL_MAP[labels[idx]] for idx in batch_indices]

            images = np.array([
                preprocess_image(path, augment=True)
                for path in batch_paths
            ])

            yield images, np.array(batch_labels)


def build_model():
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    return model


if __name__ == "__main__":
    train_paths, train_labels = load_data(TRAIN_DIR)

    model = build_model()

    history = model.fit(
        data_generator(train_paths, train_labels),
        steps_per_epoch=len(train_paths) // BATCH_SIZE,
        epochs=EPOCHS
    )

    model.save("../models/mri_tumor_model.h5")

    print("Training completed and model saved.")
