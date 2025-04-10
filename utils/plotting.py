# plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_voltage_traces(ensemble):
    num_neurons = len(ensemble.neurons)
    cols = 2
    rows = (num_neurons + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), sharex=True, sharey=True)
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

    fig.suptitle("Membrane Voltage Traces of LIF Neurons", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_tunings_over_time(gain_x_history, gain_y_history, bias_history):
    episodes = len(gain_x_history[0])
    x = np.arange(episodes)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i, history in enumerate(gain_x_history):
        axs[0].plot(x, history, label=f"N{i}")
    axs[0].set_title("X Gain Evolution")
    axs[0].set_ylabel("Gain X")
    axs[0].legend()

    for i, history in enumerate(gain_y_history):
        axs[1].plot(x, history, label=f"N{i}")
    axs[1].set_title("Y Gain Evolution")
    axs[1].set_ylabel("Gain Y")

    for i, history in enumerate(bias_history):
        axs[2].plot(x, history, label=f"N{i}")
    axs[2].set_title("Bias Evolution")
    axs[2].set_ylabel("Bias")
    axs[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()
