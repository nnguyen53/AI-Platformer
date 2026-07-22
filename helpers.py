import numpy as np
from consts import *
from maps import MAPS

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


def get_beta(frame): 
    return min(BETA_END, BETA_START + (BETA_END - BETA_START) * frame / BETA_FRAMES)

def sample_batch(total_frames, replay_buffers, priorities):
    beta = get_beta(total_frames)
    batch = []

    for lvl in MAPS:
        buffer = replay_buffers[lvl]
        if len(buffer) < SAMPLES_PER_LEVEL:
            continue

        priority_arr = np.array(priorities[lvl], dtype=np.float64) ** ALPHA
        probs = priority_arr / priority_arr.sum()

        indices = np.random.choice(len(buffer), size=SAMPLES_PER_LEVEL, p=probs, replace=True)
        weights = (len(buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max() if weights.max() > 0 else weights

        for idx, w in zip(indices, weights):
            batch.append((lvl, idx, w))

    return batch