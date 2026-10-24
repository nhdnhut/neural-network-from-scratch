import sys
import os
import numpy as np
from neural_net import NeuralNetwork
from data.get_mnist import get_mnist


def main():
    print("=" * 60)
    print("  Interpretable Neural Networks from Scratch: MNIST Training")
    print("  Author: Nguyen Huynh Dang Nhut (@nhutnhd)")
    print("=" * 60)
    
    # 1. Load MNIST data
    print("\n[INFO] Loading MNIST dataset...")
    images, labels = get_mnist()
    
    # Reshape and normalize
    # images shape: (60000, 784), labels shape: (60000, 10)
    num_train = 50000
    x_train, y_train = images[:num_train], labels[:num_train]
    x_val, y_val = images[num_train:], labels[num_train:]
    
    print(f"[INFO] Train samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}")
    
    # 2. Instantiate Network: 784 -> 64 -> 10
    model = NeuralNetwork([784, 64, 10], activation="sigmoid")
    
    epochs = 5
    batch_size = 64
    learning_rate = 0.5
    
    print(f"\n[INFO] Architecture: 784 -> 64 -> 10 | Batch Size: {batch_size} | LR: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        loss, train_acc = model.train_epoch(x_train, y_train, batch_size=batch_size, learning_rate=learning_rate)
        
        # Validation accuracy
        val_logits = model.forward(x_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1) if y_val.ndim == 2 else y_val
        val_acc = np.mean(val_preds == val_true)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    print("-" * 60)
    print("[SUCCESS] Training completed successfully!")


if __name__ == "__main__":
    main()
