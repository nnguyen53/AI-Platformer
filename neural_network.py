import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def run(self, state):
        """
        Run a set of input values through all layers, and output the column of output values
        """
        values = state
        caches = []

        for i, layer in enumerate(self.layers):
            values, cache = layer.feed_forward(
                values,
                i != len(self.layers)-1
            )

            caches.append(cache)

        return values, caches
        
    def backpropagate(self, prediction, target, caches, weight=1.0):
        """
        Backpropagates through the neural network by adjusting weights and biases according to the prediction error.
        """
        learning_rate = 0.0005

        delta = prediction - target
        delta = np.where(
            np.abs(delta) <= 1,
            delta,
            np.sign(delta)
        )

        delta = delta * weight

        delta = self.layers[-1].backward_pass(
            delta,
            learning_rate,
            caches[-1],
            output_layer = True
        )

        for layer, cache in zip(
                reversed(self.layers[:-1]),
                reversed(caches[:-1])
            ):
            delta = layer.backward_pass(delta, learning_rate, cache)
    
    def save(self, path):
        """
        Saves the current network parameters at the specified path
        """
        data = {}

        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.weights
            data[f"B{i}"] = layer.biases

        np.savez(path, **data)

    def load(self, path):
        """
        Initializes all layer parameters with the values saved at the specified path
        """
        data = np.load(path)

        for index, layer in enumerate(self.layers):
            layer.weights = data[f"W{index}"]
            layer.biases = data[f"B{index}"]

