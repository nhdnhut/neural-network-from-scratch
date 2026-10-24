# Interpretable Neural Networks from Scratch

> First-principles implementation of feedforward propagation, analytical gradient backpropagation, and finite-difference gradient checking using pure Python and NumPy.
> Author: Nguyen Huynh Dang Nhut (@nhdnhut)
> Research Logbook & Theoretical Foundation: [What Is Happening Inside a Neural Network?](https://nhdnhut.github.io/writings/05-what-is-happening-inside-a-neural-network.html)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-4%20Passing%20%28Gradient%20Checked%29-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## 1. Overview

In modern deep learning development, framework abstractions (such as PyTorch autograd or TensorFlow gradient tapes) compute gradients automatically behind a single `.backward()` call. While convenient for rapid prototyping, this layer of abstraction often conceals the exact tensor mechanics, intermediate cache buffers, and multivariate chain rule dynamics governing neural learning.

This repository rebuilds a multi-layer feedforward neural network entirely from first principles using standard Python and NumPy, validated with analytical gradient checking against numerical finite differences ($\lt 10^{-6}$ relative error).

---

## 2. Mathematical Foundations

### 2.1 Feedforward Propagation

For an input mini-batch matrix $X \in \mathbb{R}^{N \times D_{\text{in}}}$:

1. **Affine Transformation**:
   $$Z = X \cdot W + b$$
   Where $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ is the weight matrix and $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ is the bias vector broadcast across mini-batch rows.

2. **Non-Linear Activation**:
   $$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$

### 2.2 Analytical Backpropagation & The Multivariate Chain Rule

Given downstream gradient $\frac{\partial L}{\partial Z} = \delta$:

1. **Weight Gradient**:
   $$\frac{\partial L}{\partial W} = \frac{1}{N} X^T \cdot \delta$$

2. **Bias Gradient**:
   $$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} \delta_i$$

3. **Input Gradient (Upstream Pass)**:
   $$\frac{\partial L}{\partial X} = \delta \cdot W^T$$

4. **Sigmoid Activation Derivative**:
   $$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \left( \sigma(Z) \odot (1 - \sigma(Z)) \right)$$

---

## 3. Finite-Difference Numerical Gradient Checking

To rigorously verify that analytical gradient derivations are mathematically correct, the test suite executes numerical finite-difference approximations:

$$\frac{\partial L}{\partial W_{ij}} \approx \frac{L(W_{ij} + \epsilon) - L(W_{ij} - \epsilon)}{2\epsilon}$$

Relative error is evaluated via:

$$\text{Error} = \frac{\| \nabla_{\text{analytical}} - \nabla_{\text{numerical}} \|_2}{\| \nabla_{\text{analytical}} \|_2 + \| \nabla_{\text{numerical}} \|_2} \lt 10^{-6}$$

Run the verification suite:
```bash
pytest tests/ -v
```

---

## 4. Directory Structure

```text
neural-network-from-scratch/
|-- data/
|   `-- get_mnist.py          # MNIST byte parser and normalizer
|-- tests/
|   `-- test_gradients.py     # Analytical vs numerical gradient check (relative error < 1e-6)
|-- neural_net.py             # Modular DenseLayer, Sigmoid, ReLU, Softmax, NeuralNetwork
|-- main.py                   # Full MNIST training harness (784 -> 64 -> 10)
|-- pytest.ini                # Pytest path configuration
|-- LICENSE                   # MIT License
`-- README.md                 # Theoretical and operational guide
```

---

## 5. Quickstart Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/nhdnhut/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Run tests to verify gradient mechanics:
   ```bash
   pytest tests/ -v
   ```

3. Train on MNIST:
   ```bash
   python main.py
   ```

---

## 6. License

This project is licensed under the MIT License. Copyright (c) 2024-2026 Nguyen Huynh Dang Nhut (@nhdnhut).
Detailed writeup available on [nhdnhut.github.io](https://nhdnhut.github.io).

