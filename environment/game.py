import random

from environment.grid import Grid
from environment.state import build_state

from entities.player import Player
from entities.predator import Predator
from entities.token import Token
from entities.safezone import SafeZone


class ForagingGame:

    ACTIONS = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
        4: (0, 0)    # stay
    }

    def __init__(self):

        self.grid = Grid(24, 16)

        self.max_steps = 200

        self.reset()

    def reset(self):

        self.steps = 0

        self.player = Player(2, 2)

        self.safe_zone = SafeZone(0, 0)

        self.predator = Predator(
            23,
            15,
            wake_probability=0.002
        )

        self.tokens = self.generate_tokens(20)

        return build_state(self)

    def generate_tokens(self, n):

        tokens = []

        for _ in range(n):

            x = random.randint(0, 23)
            y = random.randint(0, 15)

            tokens.append(Token(x, y))

        return tokens

    def step(self, action):

        self.steps += 1

        dx, dy = self.ACTIONS[action]

        new_x = self.player.x + dx
        new_y = self.player.y + dy

        if self.grid.inside(new_x, new_y):
            self.player.move(dx, dy)

        reward = -0.01

        reward += self.collect_tokens()

        self.predator.update(self.player)

        done = False

        if self.predator.catches(self.player):

            reward -= 20
            done = True

        elif self.player_in_safe_zone():

            reward += 5
            done = True

        elif self.steps >= self.max_steps:

            done = True

        return (
            build_state(self),
            reward,
            done
        )

    def collect_tokens(self):

        collected = []

        reward = 0

        for token in self.tokens:

            if (
                token.x == self.player.x and
                token.y == self.player.y
            ):
                reward += 1
                self.player.tokens += 1
                collected.append(token)

        for token in collected:
            self.tokens.remove(token)

        return reward

    def player_in_safe_zone(self):

        return (
            self.player.x == self.safe_zone.x and
            self.player.y == self.safe_zone.y
        )