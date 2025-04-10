# learning.py

def apply_reward_modulated_hebbian_learning(ensemble, input_vector, spikes, reward, lr=0.05):
    for i in range(len(ensemble.neurons)):
        if spikes[i]:
            ensemble.gains[i][0] += lr * reward * input_vector[0]
            ensemble.gains[i][1] += lr * reward * input_vector[1]
            ensemble.biases[i] += lr * reward
