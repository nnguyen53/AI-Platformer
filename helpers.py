import numpy as np

def he_initialize(num_in, num_out):
    """
    Initializes weights and biases with the He initialization technique to account for possible signal loss with ReLu activation
    """
    weights = np.random.randn(num_out, num_in) * np.sqrt(2 / num_in)
    biases = np.zeros((num_out, 1))

    return weights, biases

def reLu(values):
    """
    ReLu activation function. Returns 0 for negative numbers, and the value for a positive number.
    """
    return np.maximum(0, values)

def listToColumn(values):
    """
    Convert a list of items to a NumPy column 
    """
    return np.array(values).reshape(-1, 1)