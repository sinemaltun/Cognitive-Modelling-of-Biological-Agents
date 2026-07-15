from entities.position import Position

class Token(Position):
    def __init__(self, x, y):
        super().__init__(x, y)