import pytest
import numpy as np
from neural_net import DenseLayer, Sigmoid, SoftmaxCrossEntropy, NeuralNetwork


def test_dense_layer_shapes():
    layer = DenseLayer(in_features=10, out_features=4)
    x = np.random.randn(5, 10)
    out = layer.forward(x)
    assert out.shape == (5, 4)
    
    grad_out = np.random.randn(5, 4)
    grad_x = layer.backward(grad_out)
    assert grad_x.shape == (5, 10)
    assert layer.grad_weights.shape == (10, 4)
    assert layer.grad_biases.shape == (1, 4)


def test_sigmoid_activation():
    sig = Sigmoid()
    x = np.array([-100.0, 0.0, 100.0])
    out = sig.forward(x)
    assert np.isclose(out[1], 0.5)
    assert np.isclose(out[0], 0.0, atol=1e-5)
    assert np.isclose(out[2], 1.0, atol=1e-5)


def test_numerical_gradient_check():
    """
    Finite-difference gradient check validating that analytical backprop
    matches numerical approximations with relative error < 1e-6.
    """
    np.random.seed(42)
    in_dim = 4
    out_dim = 2
    batch_size = 3
    
    layer = DenseLayer(in_dim, out_dim)
    loss_fn = SoftmaxCrossEntropy()
    
    x = np.random.randn(batch_size, in_dim)
    y = np.random.randint(0, out_dim, size=batch_size)
    
    # 1. Analytical gradients
    logits = layer.forward(x)
    loss = loss_fn.forward(logits, y)
    grad_logits = loss_fn.backward()
    layer.backward(grad_logits)
    analytical_grad_w = layer.grad_weights.copy()
    
    # 2. Finite difference numerical gradient for weights
    eps = 1e-5
    numerical_grad_w = np.zeros_like(layer.weights)
    for i in range(in_dim):
        for j in range(out_dim):
            w_orig = layer.weights[i, j]
            
            # Loss(w + eps)
            layer.weights[i, j] = w_orig + eps
            l_plus = loss_fn.forward(layer.forward(x), y)
            
            # Loss(w - eps)
            layer.weights[i, j] = w_orig - eps
            l_minus = loss_fn.forward(layer.forward(x), y)
            
            layer.weights[i, j] = w_orig
            numerical_grad_w[i, j] = (l_plus - l_minus) / (2.0 * eps)
            
    # Compute relative error: ||g_a - g_n|| / (||g_a|| + ||g_n||)
    diff = np.linalg.norm(analytical_grad_w - numerical_grad_w)
    norm = np.linalg.norm(analytical_grad_w) + np.linalg.norm(numerical_grad_w)
    rel_error = diff / norm
    
    assert rel_error < 1e-6, f"Gradient check failed! Relative error: {rel_error}"


def test_network_training_step():
    net = NeuralNetwork([8, 16, 2], activation="sigmoid")
    x = np.random.randn(20, 8)
    y = np.random.randint(0, 2, size=20)
    
    initial_loss = net.loss_fn.forward(net.forward(x), y)
    for _ in range(5):
        loss, acc = net.train_epoch(x, y, batch_size=10, learning_rate=0.1)
    
    final_loss = net.loss_fn.forward(net.forward(x), y)
    assert final_loss < initial_loss, "Training step failed to decrease loss."
