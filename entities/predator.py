from dataclasses import dataclass
import random


@dataclass
class Predator:
    x: int
    y: int

    wake_probability: float

    awake: bool = False

    def position(self):
        return self.x, self.y

    def update(self, player):

        if not self.awake:

            if random.random() < self.wake_probability:
                self.awake = True

            return

        self.chase(player)

    def chase(self, player):

        if self.x < player.x:
            self.x += 1
        elif self.x > player.x:
            self.x -= 1

        if self.y < player.y:
            self.y += 1
        elif self.y > player.y:
            self.y -= 1

    def catches(self, player):
        return (
            self.x == player.x and
            self.y == player.y
        )