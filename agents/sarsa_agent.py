import random
from collections import defaultdict

import numpy as np
import pickle
from pathlib import Path

from agents.base_agent import BaseAgent
from environment import Action

class SARSAAgent(BaseAgent):
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.02,
    ):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.actions = list(Action)

        self.q = defaultdict(
            lambda: np.zeros(len(self.actions))
        )

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return self.greedy_action(state)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done,
    ):
        action_index = self.actions.index(action)

        current_q = self.q[state][action_index]

        if done:
            target = reward
        else:
            next_action_index = self.actions.index(next_action)
            target = (
                reward
                + self.gamma
                * self.q[next_state][next_action_index]
            )

        self.q[state][action_index] += (
            self.alpha * (target - current_q)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def greedy_action(self, state):
        q_values = self.q[state]

        max_q = np.max(q_values)
        best_indices = np.flatnonzero(q_values == max_q)
        best_index = random.choice(best_indices)

        return self.actions[best_index]

# Training

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as file:
            pickle.dump(dict(self.q), file)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as file:
            loaded_q = pickle.load(file)

        self.q.update(loaded_q)