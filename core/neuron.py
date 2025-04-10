import numpy as np

class RLAgent:
    def __init__(self, brain, lr = 0.1, gamma = 0.9):
        self.brain = brain
        self.q_table = {} #For tracking value estimates
        self.lr = lr
        self.gamma = gamma 

    def select_action(self, state):
        key = tuple(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(4)
        spikes = self.brain.forward(state)
        if np.sum(spikes) == 0 or np.random.rand() < 0.1:
            return np.random.randint(4)
        return np.argmax(spikes)
    
    def learn(self, state, action, reward, next_state, done):
        key = tuple(state)
        next_key = tuple(next_state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(4)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(4)
        
        target = reward + self.gamma * np.max(self.q_table[next_key]) * (not done)
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])