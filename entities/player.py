from dataclasses import dataclass

from entities.position import Position


@dataclass
class Player:
    position: Position
    collected_tokens: int = 0

    def move_to(self, new_position: Position) -> None:
        self.position = new_position

    def collect_token(self) -> None:
        self.collected_tokens += 1

    def lose_tokens(self) -> None:
        self.collected_tokens = 0