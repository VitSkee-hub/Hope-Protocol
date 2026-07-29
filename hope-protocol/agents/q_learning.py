# agents/q_learning.py

import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table initialized to zeros
        self.q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ):
        best_next = np.max(self.q[next_state])
        target = reward if done else reward + self.gamma * best_next

        self.q[state, action] += self.alpha * (target - self.q[state, action])
