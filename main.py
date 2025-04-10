# main.py
from core.ensemble import Ensemble
from env.gridworld import GridWorldEnv
from core.encoding import encode_state
from core.learning import apply_reward_modulated_hebbian_learning
from core.utlis import select_action_from_spikes
from utils.plotting import plot_voltage_traces, plot_tunings_over_time

import numpy as np

if __name__ == "__main__":
    env = GridWorldEnv()
    ensemble = Ensemble(num_neurons=4)

    num_episodes = 10
    steps_per_episode = 100
    grid_size = env.grid_size

    gain_history_x = [[] for _ in range(ensemble.num_neurons)]
    gain_history_y = [[] for _ in range(ensemble.num_neurons)]
    bias_history = [[] for _ in range(ensemble.num_neurons)]

    for episode in range(num_episodes):
        state = env.reset()
        for step in range(steps_per_episode):
            input_vector = encode_state(state, grid_size)

            spikes = ensemble.step(input_vector)
            action = select_action_from_spikes(spikes)

            next_state, reward, done = env.step(action)
            apply_reward_modulated_hebbian_learning(ensemble, input_vector, spikes, reward)

            state = next_state
            if done:
                break

        for i in range(ensemble.num_neurons):
            gain_history_x[i].append(ensemble.gains[i][0])
            gain_history_y[i].append(ensemble.gains[i][1])
            bias_history[i].append(ensemble.biases[i])

    plot_voltage_traces(ensemble)
    plot_tunings_over_time(gain_history_x, gain_history_y, bias_history)
