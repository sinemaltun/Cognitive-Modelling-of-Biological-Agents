from entities.position import Position

class Predator(Position):
    def __init__(self, x=0, y=0):
        super().__init__(x, y)

    def move_towards(self, target_x, target_y):
        if self.x < target_x: self.x += 1
        elif self.x > target_x: self.x -= 1
        elif self.y < target_y: self.y += 1
        elif self.y > target_y: self.y -= 1