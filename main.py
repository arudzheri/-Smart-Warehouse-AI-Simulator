import pygame
from warehouse import WarehouseEnv
from q_learning import QLearningAgent
from config import GRID_SIZE, EPISODES
import time
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 500, 500
CELL_SIZE = WIDTH // GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Warehouse AI Simulator")

env = WarehouseEnv(grid_size=GRID_SIZE)
agent = QLearningAgent(state_size=GRID_SIZE * GRID_SIZE, action_size=4)

def draw(env):
    screen.fill((255, 255, 255))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    # Obstacles
    for (ox, oy) in env.obstacles:
        pygame.draw.rect(screen, (100, 100, 100), (oy * CELL_SIZE, ox * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Agent
    ax, ay = env.agent_pos
    pygame.draw.rect(screen, (0, 128, 255), (ay * CELL_SIZE, ax * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Goal
    gx, gy = env.goal
    pygame.draw.rect(screen, (0, 255, 0), (gy * CELL_SIZE, gx * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()

episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        draw(env)
        time.sleep(0.05)

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

agent.save_q_table()
pygame.quit()

# Plot reward graph
plt.plot(episode_rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()

