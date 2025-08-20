# MLP from scratch with numpy and keras

This repository contains three small projects demonstrating the implementation and training of Multi-Layer Perceptrons (MLPs) using both Keras and NumPy (from scratch).

## 1. MLP with One Hidden Layer (From Scratch)

Dataset: A custom regression dataset (dataset.xlsx)
Framework: Pure NumPy

Features:

* One hidden layer with sigmoid activation
* Forward pass, backpropagation, and weight updates implemented manually
* Gradient clipping to prevent exploding gradients
* Evaluation with MSE loss and R² score

Goal: Demonstrate how a basic neural network learns regression tasks without high-level libraries.

## 2. MLP with Two Hidden Layers (From Scratch)

Similar setup as Project 1, but with two hidden layers for more complex representation. With fully manual forward pass and backpropagation. Includes normalization, gradient clipping, and training/validation monitoring.

Goal: Show how deeper architectures can improve learning capacity while still being implemented entirely from scratch.

## 3. Iris dataset Classification with Keras

Tools: Keras, TensorFlow, scikit-learn, seaborn, matplotlib

Key steps:

* Data preprocessing (standardization, label encoding, one-hot encoding)
* Visualization with pairplots and KDE distributions
* Training a simple neural network (Dense layers with ReLU and Softmax)
* Evaluation using accuracy, confusion matrix, and learning curves

Goal: Classify flower species (Setosa, Versicolor, Virginica) with a feedforward neural network.



## Requirements

Install the necessary libraries before running the scripts:

pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow openpyxl

## How to Run

1. Clone the repository:

git clone https://github.com/Helya-Haji/MLP-from-scratch
cd your-repo-name

2. Run the scripts individually:

python iris_keras.py

python mlp_1hidden_numpy.py

python mlp_2hidden_numpy.py
