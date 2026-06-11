class Grid:

    def __init__(self, width=24, height=16):
        self.width = width
        self.height = height

    def inside(self, x, y):

        return (
            0 <= x < self.width and
            0 <= y < self.height
        )