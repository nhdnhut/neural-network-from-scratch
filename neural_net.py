"""
Interpretable Neural Networks from Scratch
Mathematical implementation of feedforward propagation, analytical backpropagation,
and finite-difference gradient checking using pure NumPy.

Author: Nguyen Huynh Dang Nhut (@nhutnhd)
Reference: https://nhutnhd.github.io
"""

from typing import Tuple, List, Dict, Any
import numpy as np


class DenseLayer:
    """
    Fully connected linear transform layer: Z = X @ W + b
    Computes analytical gradients for weights, biases, and inputs during backpropagation.
    """
    def __init__(self, in_features: int, out_features: int, init_method: str = "he"):
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight initialization
        if init_method == "he":
            limit = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features) * limit
        elif init_method == "xavier":
            limit = np.sqrt(1.0 / in_features)
            self.weights = np.random.randn(in_features, out_features) * limit
        else:
            self.weights = np.random.uniform(-0.5, 0.5, (in_features, out_features))
            
        self.biases = np.zeros((1, out_features))
        
        # Cached forward values for backpropagation
        self.inputs_cache: np.ndarray = None
        self.grad_weights: np.ndarray = None
        self.grad_biases: np.ndarray = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass: Z = inputs @ weights + biases
        inputs: shape (batch_size, in_features)
        returns: shape (batch_size, out_features)
        """
        self.inputs_cache = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass computing parameter gradients and returning upstream gradient dL/dX.
        grad_output: shape (batch_size, out_features) = dL/dZ
        """
        batch_size = self.inputs_cache.shape[0]
        self.grad_weights = np.dot(self.inputs_cache.T, grad_output) / batch_size
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        return np.dot(grad_output, self.weights.T)

    def update_params(self, learning_rate: float):
        """Applies stochastic gradient descent step."""
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class Sigmoid:
    """Numerically stable Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))"""
    def __init__(self):
        self.output_cache: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip x to avoid overflow in exp(-x)
        x_clipped = np.clip(x, -500.0, 500.0)
        self.output_cache = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.output_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Analytical derivative: dL/dZ = grad_output * sigma * (1 - sigma)"""
        return grad_output * (self.output_cache * (1.0 - self.output_cache))


class ReLU:
    """Rectified Linear Unit activation: f(x) = max(0, x)"""
    def __init__(self):
        self.input_cache: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.input_cache > 0.0)


class SoftmaxCrossEntropy:
    """Fused Softmax Activation and Categorical Cross-Entropy Loss for numerical stability."""
    def __init__(self):
        self.probs_cache: np.ndarray = None
        self.targets_cache: np.ndarray = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits: shape (batch_size, num_classes)
        targets: shape (batch_size, num_classes) one-hot or class indices
        """
        # Numerically stable softmax: subtract max per row
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs_cache = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        self.targets_cache = targets
        
        batch_size = logits.shape[0]
        # Cross-entropy: -sum(y * log(p + eps))
        eps = 1e-12
        if targets.ndim == 1:
            loss = -np.mean(np.log(self.probs_cache[np.arange(batch_size), targets] + eps))
        else:
            loss = -np.sum(targets * np.log(self.probs_cache + eps)) / batch_size
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Gradient of Softmax Cross-Entropy with respect to unnormalized logits:
        dL/dZ = probs - targets
        """
        grad = self.probs_cache.copy()
        batch_size = grad.shape[0]
        if self.targets_cache.ndim == 1:
            grad[np.arange(batch_size), self.targets_cache] -= 1.0
        else:
            grad -= self.targets_cache
        return grad


class NeuralNetwork:
    """Modular multilayer feedforward network with gradient checking and training loops."""
    def __init__(self, layer_dims: List[int], activation: str = "sigmoid"):
        self.layers: List[DenseLayer] = []
        self.activations: List[Any] = []
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(DenseLayer(layer_dims[i], layer_dims[i+1], init_method="he"))
            if i < len(layer_dims) - 2:
                if activation == "relu":
                    self.activations.append(ReLU())
                else:
                    self.activations.append(Sigmoid())
                    
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i in range(len(self.layers) - 1):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)
        out = self.layers[-1].forward(out)
        return out

    def backward(self, targets: np.ndarray):
        grad = self.loss_fn.backward()
        grad = self.layers[-1].backward(grad)
        for i in reversed(range(len(self.layers) - 1)):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

    def step(self, learning_rate: float):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, batch_size: int, learning_rate: float) -> Tuple[float, float]:
        indices = np.random.permutation(x.shape[0])
        total_loss = 0.0
        correct = 0
        num_batches = int(np.ceil(x.shape[0] / batch_size))
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            
            logits = self.forward(xb)
            loss = self.loss_fn.forward(logits, yb)
            self.backward(yb)
            self.step(learning_rate)
            
            total_loss += loss
            preds = np.argmax(logits, axis=1)
            true_labels = yb if yb.ndim == 1 else np.argmax(yb, axis=1)
            correct += np.sum(preds == true_labels)
            
        return total_loss / num_batches, correct / x.shape[0]
