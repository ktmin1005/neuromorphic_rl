# encoding.py
import numpy as np

def encode_state(state, grid_size):
    return np.array(state) / (grid_size - 1)
