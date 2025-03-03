import numpy as np
import random

# Simple Grid Environment
class GridWorld:
    def __init__(self):
        self.state = (0, 0)
        self.goal = (2, 2)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        x, y = max(0, min(2, x + moves[action][0])), max(0, min(2, y + moves[action][1]))
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        return self.state, reward, self.state == self.goal

# Q-learning Agent
class QLearningAgent:
    def __init__(self, env):
        self.q_table = np.zeros((3, 3, 4))
        self.env = env

    def choose_action(self, state):
        return random.randint(0, 3)

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        self.q_table[x, y, action] = reward + 0.9 * np.max(self.q_table[nx, ny])

# Training
env = GridWorld()
agent = QLearningAgent(env)
for _ in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

# Test
state = env.reset()
done = False
print("Trained Path:")
while not done:
    action = agent.choose_action(state)
    state, _, done = env.step(action)
    print(state)
