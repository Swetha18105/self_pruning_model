# Self-Pruning Neural Network (CIFAR-10)

This project implements a neural network that learns to prune its own connections during training using learnable sigmoid gates and L1 sparsity regularization.

## Features
- Custom `PrunableLinear` layer with learnable gates
- L1 sparsity loss to remove weak connections
- Trade-off between accuracy and sparsity
- Gate distribution visualization

## Model
3072 → 512 → 256 → 10

## How to Run
```bash
pip install -r requirements.txt
python train.py
