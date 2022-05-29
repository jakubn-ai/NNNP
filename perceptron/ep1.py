from typing import List
import numpy as np
from perceptron import Perceptron, Layer

np.random.seed(0)

"""
1. tyle ile mamy neuronow to mamy biasów
2. tyle ile mamy połączeń pomiędzy neuronami (każdy z każdym to tyle mamy wag) - jeśli dropout to 0 :)
"""

"""mamy na wejsciu wektor o rozmiarze (3, ) czyli potrzebujemy do jednego neuronu podpiąć 3 wagi i jak wyzej 
napisane 1 neuron = 1 bias"""


def get_output(W: np.ndarray, x: np.ndarray, b: np.ndarray):
    return [[dot_product(i, W) + b for W, b in zip(W, b)] for i in x]


def dot_product(x: np.ndarray, W: np.ndarray):
    return sum([x * y for x, y in zip(x, W)])


def numpy_get_output(W: np.ndarray, x: np.ndarray, b: np.ndarray):
    return np.dot(x, W.T) + b


def get_inputs():
    return np.array([[1, 2, 3, 2.5],
                     [2.0, 5.0, -1.0, 2.0],
                     [-1.5, 2.7, 3.3, -0.8]])


def get_weights():
    weights = np.array([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])
    return weights


def get_biases():
    return np.array([2, 3, 0.5])


if __name__ == "__main__":
    x_s = get_inputs()
    W_s = get_weights()
    b_s = get_biases()

    l = Layer(X=x_s, hidden_size=3)

    print(l())
    print(numpy_get_output(W_s, x_s, b_s))
