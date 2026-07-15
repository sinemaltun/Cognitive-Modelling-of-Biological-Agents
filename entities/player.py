from entities.position import Position

class Player(Position):
    def __init__(self, x=0, y=0):
        super().__init__(x, y)
        self.tokens = 0

    def move(self, dx, dy):
        self.x += dx
        self.y += dy