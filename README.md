# MNIST Digit Recognition Project

## Overview
This repository contains implementations of MNIST digit recognition using multiple deep learning frameworks. The project demonstrates fundamental deep learning concepts through the classic computer vision task of recognizing handwritten digits (0-9) from the MNIST dataset.

## Implementations
The repository includes three different implementations:
1. **PyTorch Implementation** - Using PyTorch's neural network modules and CNNs
2. **TensorFlow/Keras Implementation** - Using TensorFlow's high-level Keras API and CNNs
3. **From-Scratch Implementation** - Custom neural network implementation with NumPy (MLP architecture)

## Features
- Complete training and evaluation pipelines for all three frameworks
- Convolutional Neural Network (CNN) architectures in PyTorch and TensorFlow
- Simple Multi-Layer Perceptron (MLP) from scratch with NumPy
- Comprehensive performance visualization and comparison tools

## Project Structure
```
mnist-digit-recognition/
├── models/
│   ├── torch_mnist.py        # PyTorch implementation
│   ├── tensorflow_mnist.py   # TensorFlow/Keras implementation
│   ├── scratch_mnist.py      # From-scratch NumPy implementation
│   └── saved_models/         # Saved model files
├── analysis/
│   └── model_comparisons.py  # Script for comparing implementations 
├── results/                  # Visualization outputs
│   ├── comparison_results.png
│   └── batch_size_comparison.png
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
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
### Running Individual Implementations
#### PyTorch Implementation
```bash
python models/torch_mnist.py
```

#### TensorFlow Implementation
```bash
python models/tensorflow_mnist.py
```

#### From-Scratch Implementation
```bash
python models/scratch_mnist.py
```

### Comparing Implementations
To run a comprehensive comparison between all three implementations:
```bash
python analysis/model_comparisons.py
```

The comparison script analyzes:
- Training time differences
- Convergence rates
- Final accuracy
- Impact of batch size
- Architecture differences (CNN vs MLP)

## Model Architectures

### CNN Architecture (PyTorch & TensorFlow)
Both PyTorch and TensorFlow implementations use a similar CNN architecture:
- 2 Convolutional layers with ReLU activation and max pooling
- Fully connected layers with ReLU activation
- Output layer with 10 classes (digits 0-9)

Architecture details:
1. Conv2D (1→32, 3×3) + ReLU + MaxPool (2×2)
2. Conv2D (32→64, 3×3) + ReLU + MaxPool (2×2)
3. Fully Connected (7×7×64→128) + ReLU
4. Fully Connected (128→10) + Softmax

### MLP Architecture (NumPy From-Scratch)
The from-scratch implementation uses a simple feedforward neural network:
1. Input layer (784 neurons - flattened 28×28 images)
2. Hidden layer (128 neurons) with sigmoid/ReLU activation
3. Output layer (10 neurons) with softmax activation

## Performance
- **CNN models (PyTorch/TensorFlow)** achieve approximately 98-99% accuracy on the MNIST test set after just a few epochs
- **MLP model (NumPy)** achieves around 92-95% accuracy after the same number of epochs
- Training time and convergence rate comparisons are visualized in the results directory

## Implementation Comparison

### Key Differences
- **CNN vs MLP**: The CNN implementations capture spatial patterns in the images using convolutional layers, while the MLP treats each pixel as an independent feature
- **Parameter Count**: The CNN models have more parameters but capture the structure of the digits more effectively
- **Framework Overhead**: PyTorch and TensorFlow add some overhead but provide optimized operations and automatic differentiation
- **Training Speed**: The framework implementations typically train faster due to optimized operations and GPU support

### Comparison Results
The `analysis/model_comparisons.py` script generates comparison visualizations showing:
1. Training loss curves
2. Test accuracy over epochs
3. Total training time
4. Final test accuracy
5. Batch size impact analysis

## Visualization Examples
The code generates visualizations for:
- Training and validation loss curves for each implementation
- Accuracy progression over epochs
- Sample predictions with correct/incorrect classification highlighting
- Framework performance comparisons

## Acknowledgments
- The MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
- PyTorch and TensorFlow documentation

---

This project showcases practical deep learning skills including:
- Neural network architecture design and comparison (CNN vs MLP)
- Training pipeline implementation in multiple frameworks
- Model evaluation techniques
- Framework proficiency (PyTorch, TensorFlow, NumPy)
- Performance analysis and optimization