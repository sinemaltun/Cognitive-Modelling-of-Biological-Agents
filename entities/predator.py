from dataclasses import dataclass

from entities.position import Position


@dataclass
class Predator:
    position: Position
    awake: bool = False
    speed: int = 2

    def wake_up(self) -> None:
        self.awake = True

    def put_to_sleep(self) -> None:
        self.awake = False

    def move_towards(self, target: Position) -> None:
        """
        Move one grid cell toward the target using Manhattan pursuit.
        Prioritizes horizontal movement first, then vertical movement.
        """

        x = self.position.x
        y = self.position.y

        if x < target.x:
            x += 1
        elif x > target.x:
            x -= 1
        elif y < target.y:
            y += 1
        elif y > target.y:
            y -= 1

        self.position = Position(x, y)

    def catches(self, player_position: Position, safe_zone_position: Position) -> bool:
        """
        Predator catches the player only if they share a position
        and the player is not inside the safe zone.
        """

        return (
            self.position == player_position
            and player_position != safe_zone_position
        )