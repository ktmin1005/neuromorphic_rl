# utils.py
import numpy as np

def select_action_from_spikes(spike_vector):
    if np.sum(spike_vector) == 0:
        return np.random.randint(len(spike_vector))
    return np.argmax(spike_vector)
