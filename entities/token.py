from dataclasses import dataclass


@dataclass
class Token:
    x: int
    y: int

    def position(self):
        return self.x, self.y