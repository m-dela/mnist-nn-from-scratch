# MNIST Neural Network: A Manual Implementation

## Overview
This project is a from-scratch implementation of a Multi-Layer Perceptron (MLP) designed to classify handwritten digits trained using the MNIST dataset.

As an engineering student, the primary objective of this project was to understand the underlying mechanics of deep learning. While modern frameworks provide abstractions for automatic differentiation and layer construction, this project intentionally avoids them. PyTorch is used strictly as a linear algebra engine (for tensor operations) and for dataset loading. The forward pass, the loss function, and the full backward pass (calculating gradients via the chain rule) are derived and implemented manually.