import random

from agents.base_agent import BaseAgent
from environment import Action


class RandomAgent(BaseAgent):
    def choose_action(self, state=None):
        return random.choice(list(Action))