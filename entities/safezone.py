from dataclasses import dataclass


@dataclass
class SafeZone:
    x: int
    y: int

    def position(self):
        return self.x, self.y