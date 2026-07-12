from helpers import he_initialize, reLu
import numpy as np

class Layer:
    def __init__(self, num_in, num_out):
        """
        Creates a neural network layer with the specified number of inputs and the specified number of outputs (nodes/neurons)
        """
        self.weights, self.biases = he_initialize(num_in, num_out)

    def feed_forward(self, inputs, activate = True):
        """
        Sum all weights x inputs, and add biases. Store the inputs, weighted sum, and the result of the activation function on the weighted sum in the object.
        """
        weighted_sum = self.weights @ inputs + self.biases

        if activate:
            activation = reLu(weighted_sum)
        else:
            activation = weighted_sum

        cache = (inputs, weighted_sum, activation)

        return activation, cache
    
    def backward_pass(self, delta, learning_rate, cache, output_layer=False):
        if not output_layer:
            delta *= (cache[1] > 0).astype(float) # Account for the activation function's effect on outputs in non-output layers


        previous_delta = self.weights.T @ delta # Compute the error specifically for each of the neurons in the previous layer
        weight_change = np.clip(delta @ cache[0].T, -1, 1) 
        bias_change = np.clip(delta, -1, 1)

        self.weights -= learning_rate * weight_change # Adjust weights and biases according to the calculated shift required
        self.biases -= learning_rate * bias_change

        return previous_delta # use the error for each weight as delta for the previous layer