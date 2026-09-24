# MNIST Neural Network: A Manual Implementation

Ever wondered what happens inside the black box of ML frameworks?

This project implements a fully connected neural network from scratch - no high level machine learning engines, just pure mathematics and vector operations.
The model achieves **97.97% test accuracy** on MNIST and features and interactive web demo where you can draw your own digits and have the model guess in real time.

👉 [**Try the Live Web Demo**](https://m-dela.github.io/mnist-nn-from-scratch/)

![NN MNIST Implementation](assets/Web%20App.gif)

## Overview
After regularly using modern ML systems, I became very curious to discover their functioning - and the best way to learn was to build one from the ground up. While modern frameworks like PyTorch and TensorFlow make training a model accessible in just a few lines of code, they abstract away the core mechanics. To truly understand how deep learning works, I wanted to leave those abstractions behing and implement the underlying linear algebra, calculus, and optimization algorithm entirely by hand.

This project implements a multi-layer perceptron (MLP) trained to classify handwritten digits (0-9) using the MNIST dataset - a gallery of 60,000 28x28 greyscale images.

## Model Architecture
* **Input layer:** $784$ units ($28 \times 28$ flattened pixels, normalized to $[0,1]$)
* **Hidden layer:** $128$ units with **ReLU** activation
* **Output layer:** $10$ units with **Softmax** activation
* **Loss Function:** Categorical Cross-Entropy
* **Optimization:** Mini-batch Gradient Descend (Batch size: $m = 64$), reducing learning rate in the last epoch.

## Training Performance & Visualizations
**1. Training & Validation Loss**

![Loss Curve](assets/Loss%20over%20Time.png)
Reached **97.97%** test accuracy in 10 epochs.

**2. Error Analysis & Confusion Matrix**

![Confusion Matrix](assets/Confusion%20Matrix.png)

The confusion matrix highlights strong diagonal concentration across all 10 classes, confirming reliable performance across the board. However, examining the off-diagonal errors reveals interesting edge cases:

* **Primary Confusion (5 vs. 3):** The single most frequent error occurs with true **5s predicted as 3s (13 instances)**. This is a common failure mode in MNIST, as handwriting variations with a rounded top stroke or disconnected upper bar on a 5 closely mirror the curvature of a 3.
* **Secondary Ambiguities (4 vs. 9):**
  * **4 predicted as 9 (7 instances)**
  * **9 predicted as 4 (6 instances)**
  * This mutual confusion typically stems from whether the top vertical stroke of a 4 is closed or open, which can easily look like the closed upper loop of a 9.
* **Other Minor Discrepancies:**
  * **5 misclassified as 6 (6 instances):** Occurs when the lower loop of a 5 is drawn tightly or loops inward.
  * **7 misclassified as 9 (5 instances) & 3 misclassified as 8 (5 instances):** Minor confusions influenced by stroke angle and central curvature.
* **Most Distinct Classes:** Digits **1** (625 correct) and **7** (655 correct) achieve the highest classification precision, exhibiting almost no false positives or false negatives due to their distinct linear geometries.

![Misclassified Examples](assets/Misclassified%20Examples.png)

## Interactive Web Application

To put the model to the test outside of static test sets, I developed a lightweight web application that allows users to draw digits on an interactive HTML5 canvas and see the neural network's predictions in real time.

👉 [**Open the Live Web Demo**](https://m-dela.github.io/mnist-nn-from-scratch/)

### How It Works Under the Hood
Testing custom drawings introduces a significant domain gap between human canvas sketches and the original MNIST dataset:

1. **Bounding Box Crop & Scale:** The canvas is scanned to detect the bounding box of the drawn digit. The detected area is cropped and scaled down to fit within a $20 \times 20$ pixel frame while preserving its aspect ratio.
2. **Center-of-Mass Alignment:** The $20 \times 20$ digit is placed onto a final $28 \times 28$ grid. Its center of mass is computed across all pixel intensities, and the digit is translated so that its center of mass aligns exactly with the center of the $28 \times 28$ canvas—matching the canonical MNIST standard.
3. **Data Extraction:** The resulting canvas is converted into a 784-element array extracting each pixel's alpha (opacity) value, normalized between $[0, 1]$.
4. **Client-Side Forward Pass:** The forward pass is calculated entirely in vanilla JavaScript right inside the browser. It manually replicates the network's matrix operations using the imported weights and biases saved from training, computing the activations and output probabilities in real time without any external libraries or backend API calls.
5. **Real-Time Confidence Distribution:** Displays the probability breakdown across all 10 digits (0–9), offering direct insight into which numbers the model considers when strokes are ambiguous.
