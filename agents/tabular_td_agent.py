import random
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from environment import Action


class TabularTDAgent(BaseAgent):
    def __init__(
        self,
        alpha: float = 0.1, # learning rate
        gamma: float = 0.95, # discount factor
        epsilon: float = 0.9, # exploration
        epsilon_decay: float = 0.9999, # decay of exploration
        min_epsilon: float = 0.02, # baseline of exploration
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

    @property
    def q_table_size(self) -> int:
        return len(self.q)

    def metadata(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table_states": self.q_table_size,
        }

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return self.greedy_action(state)

    def greedy_action(self, state):
        q_values = self.q[state]

        max_q = np.max(q_values)
        best_indices = np.flatnonzero(q_values == max_q)
        best_index = random.choice(best_indices)

        return self.actions[best_index]

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as file:
            pickle.dump(dict(self.q), file)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as file:
            loaded_q = pickle.load(file)

        self.q.update(loaded_q)

    def action_index(self, action: Action) -> int:
        return self.actions.index(action)