import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageEnhance

IMAGE_SIZE = 128


def augment_image(image):
    """Apply brightness, contrast and sharpness augmentation."""

    image = Image.fromarray(np.uint8(image))

    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.8, 1.2)
    )

    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(0.8, 1.2)
    )

    image = ImageEnhance.Sharpness(image).enhance(
        random.uniform(0.8, 1.2)
    )

    return np.array(image)


def preprocess_image(image_path, augment=False):
    """Load, resize and normalize an MRI image."""

    image = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    image = img_to_array(image).astype(np.uint8)

    if augment:
        image = augment_image(image)

    image = image / 255.0

    return image
