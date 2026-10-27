# Neural Network from Scratch: Digit Recognition

A pure Python implementation of a Multi-Layer Perceptron (MLP) for handwritten digit recognition, built entirely from the ground up using **only NumPy**. 

This project was developed to deeply understand the underlying mathematics of deep learning, strictly avoiding high-level machine learning frameworks like PyTorch or TensorFlow. 

## Mathematical & Engineering Highlights

* **Forward Propagation:** Fully vectorized matrix and vector operations utilizing NumPy to ensure efficient layer-to-layer transformations.
* **Custom Backpropagation:** Manual mathematical derivation and implementation of the calculus chain rule to accurately compute gradients for all weights and biases.
* **Optimization Engine:** Implemented Gradient Descent optimization from scratch to iteratively minimize the loss function over multiple epochs.
* **Activation Functions & Derivatives:** Custom implementations of essential non-linear activations (such as ReLU and Softmax) alongside their mathematical derivatives for the backward pass.

## Repository Structure

* `main.py`: The core entry point containing the neural network class, training loop (epochs, forward/backward passes), and accuracy evaluation metrics.
* `archive/`: Directory dedicated to storing the raw dataset (e.g., MNIST CSV/binary files) and archived model weights/parameters.

## Tech Stack
* **Language:** Python
* **Libraries:** NumPy (Strictly limited to linear algebra and matrix operations)
