from dataclasses import dataclass

from entities.position import Position


@dataclass(frozen=True)
class Token:
    position: Position