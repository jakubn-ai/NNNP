from abc import ABC

import numpy as np

np.random.seed(0)


class Perceptron:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim)
        self.b = np.random.randn(1)

    def __call__(self, *args, **kwargs):
        return self.W, self.b


class LayerDense:
    def __init__(self, n_features, output_dim):
        self.params = np.array([np.concatenate(Perceptron(input_dim=n_features)()) for _ in range(output_dim)])

    @property
    def W(self) -> np.ndarray:
        return self.params[:, :-1]

    @property
    def b(self) -> np.ndarray:
        return self.params[:, -1]

    def __call__(self, X):
        return np.dot(X, self.W.T) + self.b


class Activation:
    def __call__(self, X):
        raise NotImplementedError


class ReLU(Activation):
    def __call__(self, X):
        return np.maximum(0, X)


class Softmax(Activation):
    def __call__(self, X):
        exponential_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        return probabilities
