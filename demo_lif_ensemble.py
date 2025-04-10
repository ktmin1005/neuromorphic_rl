import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau = 20.0, v_thresh = 1.0, v_reset = 0.0):
        self.v = 0.0
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_history = []
    
    def step(self, input_current, dt = 1.0, noise_prob = 0.01):
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
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        self.gains = np.random.uniform(0.5, 2.0, size = (num_neurons, 2)) #how much the neuron responds to input
        self.biases = np.random.uniform(-0.5, 0.5, size = num_neurons) #how easily it spikes

    def step(self, input_vector, timestep=None):
        spikes = []
        for i, neuron in enumerate(self.neurons):
            # Weighted sum: gain_x * x + gain_y * y + bias
            effective_input =  (
                self.gains[i][0] * input_vector[0] + 
                self.gains[i][1] * input_vector[1] +
                self.biases[i]
            )
            spike = neuron.step(effective_input)
            spikes.append(spike)
        return spikes

    
    def apply_reward_learning(self, input_vector, spikes, reward, lr = 0.05):
        for i in range(len(self.neurons)):
            if spikes[i]: # Only update neurons that spiked
                # Hebbian update modulated by reward
                self.gains[i][0] += lr * reward * input_vector[0]
                self.gains[i][1] += lr * reward * input_vector[1]
                self.biases[i] += lr * reward # Small boost if spiking leads to reward
    
class GridWorldEnv:
    def __init__(self, grid_size = 5):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)

    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)

        self.state = (x, y)
        reward = 1.0 if self.state == self.goal else -0.01
        done = self.state == self.goal
        return self.state, reward, done
    
def run_simulation(ensemble, input_signal, timesteps = 100):
    spike_data = []
    for t in range(timesteps):
        current = input_signal(t)
        spikes = ensemble.step(current)
        spike_data.append(spikes)
    return np.array(spike_data)

def input_signal(t):
    return 1.2 + 0.8 * np.sin(t / 10.0)

def select_action_from_spikes(spike_vector):
    '''
    Takes a vector of spikes (0s and 1s), one per action neuron.
    Returns the index of the action to take
    '''
    if np.sum(spike_vector) == 0:
        # No neurons spiked - choose randomly (exploration or hesitation)
        return np.random.randint(len(spike_vector))
    # Otherwise, pick the neuron that spiked (or the most active one if multiple)
    return np.argmax(spike_vector)

def encode_state(state, grid_size):
    return np.array(state) / (grid_size - 1)

def plot_voltage_traces(ensemble):
    num_neurons = len(ensemble.neurons)
    cols = 2
    rows = (num_neurons + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize = (10, 2.5 * rows), sharex = True, sharey = True)
    axs = axs.flatten()
    for i, neuron in enumerate(ensemble.neurons):
        axs[i].plot(neuron.v_history)
        axs[i].set_title(f"Neuron {i}")
        axs[i].set_ylabel("Voltage")
        axs[i].grid(True)
    for ax in axs[-cols:]:
        ax.set_xlabel("Time (steps)")
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    fig.suptitle("Membrane Voltage Traces of LIF Neurons", fontsize = 14)
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.show()

def print_tunings(ensemble):
    print("\nNeuron Tunings:")
    for i in range(len(ensemble.neurons)):
        print(f"Neuron {i}: gain = {ensemble.gains[i]:.4f}, bias = {ensemble.biases[i]:.4f}")

def plot_tunings_over_time(gain_x_history, gain_y_history, bias_history):
    episodes = len(gain_x_history[0])
    x = np.arange(episodes)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, history in enumerate(gain_x_history):
        axs[0].plot(x, history, label=f"Neuron {i}")
    axs[0].set_title("Gain X Evolution Over Episodes")
    axs[0].set_ylabel("Gain X")
    axs[0].legend()

    for i, history in enumerate(gain_y_history):
        axs[1].plot(x, history, label=f"Neuron {i}")
    axs[1].set_title("Gain Y Evolution Over Episodes")
    axs[1].set_ylabel("Gain Y")

    for i, history in enumerate(bias_history):
        axs[2].plot(x, history, label=f"Neuron {i}")
    axs[2].set_title("Bias Evolution Over Episodes")
    axs[2].set_ylabel("Bias")
    axs[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorldEnv()
    ensemble = Ensemble(num_neurons=4)

    num_episodes = 10
    steps_per_episode = 100
    grid_size = env.grid_size

    gain_x_history = [[] for _ in range(len(ensemble.neurons))]
    gain_y_history = [[] for _ in range(len(ensemble.neurons))]
    bias_history = [[] for _ in range(ensemble.biases.size)]

    for episode in range(num_episodes):
        state = env.reset()
        for step in range(steps_per_episode):
            encoded = encode_state(state, grid_size)
            input_vector = encode_state(state, grid_size)

            spikes = ensemble.step(input_vector)  # no logging
            action = select_action_from_spikes(spikes)

            next_state, reward, done = env.step(action)
            ensemble.apply_reward_learning(input_vector, spikes, reward)

            state = next_state
            if done:
                break

        for i in range(len(ensemble.neurons)):
            gain_x_history[i].append(ensemble.gains[i][0])
            gain_y_history[i].append(ensemble.gains[i][1])
            bias_history[i].append(ensemble.biases[i])

    plot_voltage_traces(ensemble)
    plot_tunings_over_time(gain_x_history, gain_y_history, bias_history)
