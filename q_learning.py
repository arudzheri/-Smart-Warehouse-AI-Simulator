import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

