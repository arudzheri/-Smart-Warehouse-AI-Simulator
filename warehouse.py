import numpy as np

class WarehouseEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.obstacles = {(3, 3), (4, 4), (5, 5), (6, 3)}  # Obstacles
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal = [self.grid_size - 1, self.grid_size - 1]
        return self.get_state()

    def get_state(self):
        return self.agent_pos[1] + self.grid_size * self.agent_pos[0]

    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and y > 0: y -= 1  # Left
        elif action == 1 and y < self.grid_size - 1: y += 1  # Right
        elif action == 2 and x > 0: x -= 1  # Up
        elif action == 3 and x < self.grid_size - 1: x += 1  # Down

        new_pos = [x, y]
        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos

        done = self.agent_pos == self.goal
        reward = 10 if done else -1
        return self.get_state(), reward, done

