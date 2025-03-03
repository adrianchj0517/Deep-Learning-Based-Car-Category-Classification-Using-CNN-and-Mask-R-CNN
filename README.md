# Car Category Classification

## 📌 Overview
This project implements a **Deep Learning-based Car Category Classification** system using **Convolutional Neural Networks (CNNs)** and **Mask R-CNN**. The model is designed to classify different car categories based on images, leveraging advanced deep learning techniques for object detection and classification.

## 🔥 Features
- **Car Image Classification**: Identifies and classifies cars into different categories.
- **Mask R-CNN for Object Detection**: Enhances the classification by segmenting car objects from the background.
- **CNN-based Model**: Utilizes deep learning for feature extraction and classification.
- **Dataset Preprocessing**: Handles data augmentation, normalization, and resizing for improved model performance.
- **Performance Metrics**: Evaluates the model using accuracy, precision, recall, and F1-score.
- **Visualization**: Displays classification results with confidence scores.

## 📊 Dataset Preparation
1. **Data Collection**: The dataset consists of car images labeled with different categories.
2. **Preprocessing**:
   - Resize images to match the input shape of the CNN.
   - Normalize pixel values for better convergence.
   - Apply data augmentation techniques (e.g., flipping, rotation, brightness adjustments).
3. **Splitting**: Divide the dataset into training, validation, and test sets.

## 🏗 Model Architecture
The project implements the following architecture:
- **Convolutional Layers**: Extracts spatial features from images.
- **Pooling Layers**: Reduces dimensionality and retains essential features.
- **Fully Connected Layers**: Performs classification based on extracted features.
- **Mask R-CNN**: Enhances the model by detecting and segmenting car objects.

## 🚀 Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (optional for running the notebook)


