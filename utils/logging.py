# logging.py

def log_step_info(step, state, action, reward):
    print(f"Step {step}: State = {state}, Action = {action}, Reward = {reward}")

def log_episode_result(episode, reached_goal):
    if reached_goal:
        print(f"Episode {episode + 1}: Goal reached! 🎯")
    else:
        print(f"Episode {episode + 1}: Did not reach goal.")

def log_neuron_parameters(ensemble):
    print("\nNeuron Tunings:")
    for i in range(len(ensemble.neurons)):
        print(f"Neuron {i}: gain = {ensemble.gains[i]}, bias = {ensemble.biases[i]:.4f}")
