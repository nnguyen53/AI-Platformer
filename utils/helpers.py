import numpy as np
import json
from utils.consts import *
from game.maps import MAPS

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


def get_beta(frame, agent_config): 
    return min(agent_config.BETA_END, 
               agent_config.BETA_START + (agent_config.BETA_END - agent_config.BETA_START) * frame / agent_config.BETA_FRAMES)

def sample_batch(total_frames, replay_buffer, priorities, agent_config):
    beta = get_beta(total_frames, agent_config)
    batch = []

    priority_arr = np.array(priorities, dtype=np.float64) ** agent_config.ALPHA
    probs = priority_arr / priority_arr.sum()

    indices = np.random.choice(len(replay_buffer), size=agent_config.SAMPLES_PER_BATCH, p=probs, replace=True)
    weights = (len(replay_buffer) * probs[indices]) ** (-beta)
    weights = weights / weights.max() if weights.max() > 0 else weights

    for idx, w in zip(indices, weights):
        batch.append((idx, w))

    return batch

def save_results(env, path="results.json"):  
    """Persists per-level win/loss results - pulled out of __main__ so both a
    hard quit and a quit-to-menu can save training progress"""
    with open(path, "w") as file:
        json.dump(env.level_results, file, indent=4)

def draw_multiline_text(surface, text, font, color, pos):
    x, y = pos
    lines = text.split('\n')
    
    for line in lines:
        line_surface = font.render(line, True, color)
        surface.blit(line_surface, (x, y))
        y += line_surface.get_height() + 5