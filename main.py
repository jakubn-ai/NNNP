import perceptron as p
import numpy as np


def get_weights():
    weights = np.array([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])
    return weights


def get_biases():
    return np.array([2, 3, 0.5])


if __name__ == '__main__':

    inputs = np.random.randn(5, 5)
    weights = get_weights()
    biases = get_biases()

    layer = p.LayerDense(n_features=inputs.shape[1], output_dim=3)
    layer2 = p.LayerDense(n_features=3, output_dim=2)
    relu = p.ReLU()
    softmax = p.Softmax()

    layer_output = layer(inputs)
    layer2_output = softmax(layer2(relu(layer_output))) # chain :)
    print(layer_output)
    print(layer2_output)

