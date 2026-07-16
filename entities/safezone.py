from dataclasses import dataclass

from entities.position import Position


@dataclass(frozen=True)
class SafeZone:
    position: Position

    def contains(self, position: Position) -> bool:
        return self.position == position