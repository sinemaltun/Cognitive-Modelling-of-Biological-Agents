from entities import Position


class Grid:
    def __init__(self, width: int = 24, height: int = 16):
        self.width = width
        self.height = height

    def inside(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height