import numpy as np

class SimpleEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)     # Up
        elif action == 1: x = min(self.grid_size - 1, x + 1)  # Down
        elif action == 2: y = max(0, y - 1)     # Left
        elif action == 3: y = min(self.grid_size - 1, y + 1)  # Right
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.01
        done = self.state == self.goal
        return self.state, reward, done

    def get_state_space(self):
        return self.grid_size * self.grid_size

    def get_action_space(self):
        return 4  # Up, Down, Left, Right