# MRI Tumor Detection System

## Overview

This project implements a deep learning-based system for classifying brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. The model uses transfer learning with VGG16, pretrained on ImageNet, followed by additional fully connected layers for classification.

The project focuses on image preprocessing, data augmentation, transfer learning, model training, and evaluation using TensorFlow and Keras.

## Classification Categories

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Model Architecture

The classification model is built using VGG16 as the feature extraction backbone.

The pretrained convolutional base is used to extract features from MRI images, while additional layers perform the final classification.

**Architecture:**

- VGG16 (pretrained on ImageNet)
- Flatten layer
- Dropout (0.3)
- Dense layer (128 neurons, ReLU)
- Dropout (0.2)
- Output layer (4 neurons, Softmax)

The initial layers of VGG16 are frozen, while the final three layers are allowed to train.

## Data Preprocessing

MRI images undergo the following preprocessing steps before being passed to the model:

- Resizing images to 128 × 128 pixels
- Conversion to numerical arrays
- Normalization of pixel values to the range 0–1
- Random brightness, contrast, and sharpness adjustments during training

Data augmentation is applied during training to introduce image variations and improve model generalization.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | VGG16 |
| Pretrained Weights | ImageNet |
| Input Size | 128 × 128 × 3 |
| Batch Size | 20 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Sparse Categorical Crossentropy |
| Output Activation | Softmax |
| Number of Classes | 4 |

## Project Structure

```text
MRI-Tumor-Detection-System/
│
├── dataset/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow (PIL)

## Workflow

1. Load the MRI image dataset.
2. Preprocess and resize images.
3. Apply data augmentation to training images.
4. Extract features using pretrained VGG16.
5. Train the classification layers.
6. Evaluate the model using test data.
7. Generate predictions for individual MRI images.

## Model Evaluation

The model evaluation process includes:

- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics are used to examine the classification performance across all four categories.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MRI-Tumor-Detection-System.git
cd MRI-Tumor-Detection-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository. Organize the downloaded dataset into the following structure:

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Usage

Training:

```bash
cd src
python train.py
```

Evaluation:

```bash
python evaluate.py
```

Prediction for a single MRI image:

```bash
python predict.py path/to/image.jpg
```

## Disclaimer

This project is developed for educational and research purposes. The model is not intended to replace professional medical diagnosis, clinical evaluation, or the advice of qualified healthcare professionals.
