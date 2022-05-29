import numpy as np

np.random.seed(0)


# weights = np.array([[0.2, 0.8, -0.5, 1.0],
#                    [0.5, -0.91, 0.26, -0.5],
#                    [-0.26, -0.27, 0.17, 0.87]])
# biases = np.array([2, 3, 0.5])

class Perceptron:
    def __init__(self, X):
        self.X = X
        self.W = np.random.randn(self.X.shape[1])
        self.b = np.random.randn(self.X.shape[0])

    def __call__(self, *args, **kwargs):
        return np.dot(self.X, self.W.T) + self.b


class Layer:
    def __init__(self, X, hidden_size):
        self.neurons = [Perceptron(X) for _ in range(hidden_size)]

    def __call__(self, *args, **kwargs):
        return np.array([p() for p in self.neurons]).T


class Dense:
    def __init__(self):
        pass
