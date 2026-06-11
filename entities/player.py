from dataclasses import dataclass


@dataclass
class Player:
    x: int
    y: int
    tokens: int = 0

    def position(self):
        return self.x, self.y

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy