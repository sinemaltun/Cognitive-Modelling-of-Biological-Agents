from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def moved(self, dx: int, dy: int) -> "Position":
        return Position(
            self.x + dx,
            self.y + dy
        )

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y