# MNIST Digit Recognition Project

## Overview
This repository contains implementations of MNIST digit recognition using multiple deep learning frameworks. The project demonstrates fundamental deep learning concepts through the classic computer vision task recognizing handwritten digits (0-9) from the MNIST dataset.

## Implementations
The repository includes three different implementations:
1. **PyTorch Implementation** - Using PyTorch's neural network modules
2. **TensorFlow/Keras Implementation** - Using TensorFlow's high-level Keras API
3. **From-Scratch Implementation** - Custom neural network implementation with NumPy

## Features
- Complete training and evaluation pipelines
- Convolutional Neural Network architectures
- Comprehensive performance visualization

## Project Structure
```
mnist-digit-recognition/
├── models/
├──── models/               # Saved model files
├──── torch_mnist.py        # PyTorch implementation
├──── tensorflow_mnist.py   # TensorFlow/Keras implementation
├──── scratch_mnist.py      # From-scratch implementation
├── requirements.txt        # Project dependencies
├── results/                # Visualization outputs
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage
### PyTorch Implementation
```bash
python torch_mnist.py
```

### TensorFlow Implementation
```bash
python tensorflow_mnist.py
```

### From-Scratch Implementation
```bash
python scratch_mnist.py
```

## Model Architecture
The CNN architecture used in both PyTorch and TensorFlow implementations consists of:
- 2 Convolutional layers with ReLU activation and max pooling
- Fully connected layers with ReLU activation
- Output layer with 10 classes (digits 0-9)

Architecture details:
1. Conv2D (1→32, 3×3) + ReLU + MaxPool (2×2)
2. Conv2D (32→64, 3×3) + ReLU + MaxPool (2×2)
3. Fully Connected (7×7×64→128) + ReLU
4. Fully Connected (128→10) + Softmax

## Performance
The models achieve approximately 98-99% accuracy on the MNIST test set after just a few epochs of training, demonstrating the effectiveness of CNNs for this classification task.

## Visualizations
The code generates visualizations for:
- Training and validation loss curves
- Training and validation accuracy curves
- Sample predictions with correct/incorrect classification highlighting

## Acknowledgments
- The MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
- PyTorch and TensorFlow documentation

---

This project showcases practical deep learning skills including:
- Neural network architecture design
- Training pipeline implementation
- Model evaluation techniques
- Framework proficiency (PyTorch, TensorFlow)