from .lif_neuron import LIFNeuron
import numpy as np

class BasalGangliaModel:
    def __init__(self, input_size=2, output_size=4):
        self.neurons = [LIFNeuron() for _ in range(output_size)]
        self.weights = np.random.randn(output_size, input_size) * 0.1

    def forward(self, state):
        state_vec = np.array(state) / 5.0  # Normalize input
        spikes = []
        for i, neuron in enumerate(self.neurons):
            input_current = np.dot(self.weights[i], state_vec)
            spike = neuron.step(input_current)
            spikes.append(spike)
        return np.array(spikes)
