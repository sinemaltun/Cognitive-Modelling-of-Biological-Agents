from entities.position import Position

class SafeZone(Position):
    def __init__(self, x, y):
        super().__init__(x, y)