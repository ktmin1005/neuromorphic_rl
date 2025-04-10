# ensemble.py
import numpy as np


class LIFNeuron:
    def __init__(self, tau=20.0, v_thresh=1.0, v_reset=0.0):
        self.v = 0.0
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_history = []

    def step(self, input_current, dt=1.0, noise_prob=0.01):
        dv = (-self.v + input_current) / self.tau
        self.v += dv * dt
        self.v_history.append(self.v)
        spike = int(self.v >= self.v_thresh)
        if not spike and np.random.rand() < noise_prob:
            spike = 1
        if spike:
            self.v = self.v_reset
        return spike


class Ensemble:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        self.gains = np.random.uniform(0.5, 2.0, size=(num_neurons, 2))
        self.biases = np.random.uniform(-0.5, 0.5, size=num_neurons)

    def step(self, input_vector, timestep=None):
        spikes = []
        for i, neuron in enumerate(self.neurons):
            effective_input = (
                self.gains[i][0] * input_vector[0] +
                self.gains[i][1] * input_vector[1] +
                self.biases[i]
            )
            spike = neuron.step(effective_input)
            spikes.append(spike)
        return spikes

    def apply_reward_learning(self, input_vector, spikes, reward, lr=0.05):
        for i in range(self.num_neurons):
            if spikes[i]:
                self.gains[i][0] += lr * reward * input_vector[0]
                self.gains[i][1] += lr * reward * input_vector[1]
                self.biases[i] += lr * reward
