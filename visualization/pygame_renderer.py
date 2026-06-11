import pygame


CELL_SIZE = 32

WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)

GREEN = (0, 180, 0)

YELLOW = (255, 220, 0)

GRAY = (140, 140, 140)
RED = (220, 0, 0)

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)


class Renderer:

    def __init__(self, env):

        pygame.init()

        self.env = env

        self.width = env.grid.width * CELL_SIZE
        self.height = env.grid.height * CELL_SIZE

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption(
            "Foraging Task"
        )

        self.font = pygame.font.SysFont(
            None,
            28
        )

    def draw(self):

        self.screen.fill(WHITE)

        self.draw_grid()

        self.draw_safe_zone()

        self.draw_tokens()

        self.draw_predator()

        self.draw_player()

        self.draw_frame()

        self.draw_score()

        pygame.display.flip()

    def draw_grid(self):

        for x in range(self.env.grid.width):

            for y in range(self.env.grid.height):

                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                pygame.draw.rect(
                    self.screen,
                    LIGHT_GRAY,
                    rect,
                    1
                )

    def draw_player(self):

        x = self.env.player.x * CELL_SIZE
        y = self.env.player.y * CELL_SIZE

        points = [
            (x + CELL_SIZE // 2, y + 4),
            (x + 4, y + CELL_SIZE - 4),
            (x + CELL_SIZE - 4, y + CELL_SIZE - 4)
        ]

        pygame.draw.polygon(
            self.screen,
            GREEN,
            points
        )

    def draw_tokens(self):

        for token in self.env.tokens:

            cx = (
                token.x * CELL_SIZE
                + CELL_SIZE // 2
            )

            cy = (
                token.y * CELL_SIZE
                + CELL_SIZE // 2
            )

            diamond = [
                (cx, cy - 8),
                (cx + 8, cy),
                (cx, cy + 8),
                (cx - 8, cy)
            ]

            pygame.draw.polygon(
                self.screen,
                YELLOW,
                diamond
            )

    def draw_predator(self):

        color = (
            RED
            if self.env.predator.awake
            else GRAY
        )

        pygame.draw.circle(
            self.screen,
            color,
            (
                self.env.predator.x * CELL_SIZE
                + CELL_SIZE // 2,

                self.env.predator.y * CELL_SIZE
                + CELL_SIZE // 2
            ),
            CELL_SIZE // 3
        )

    def draw_safe_zone(self):

        pygame.draw.rect(
            self.screen,
            BLACK,
            (
                self.env.safe_zone.x * CELL_SIZE,
                self.env.safe_zone.y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

    def draw_frame(self):

        color = (
            RED
            if self.env.predator.awake
            else BLUE
        )

        pygame.draw.rect(
            self.screen,
            color,
            self.screen.get_rect(),
            width=6
        )

    def draw_score(self):

        text = self.font.render(
            f"Tokens: {self.env.player.tokens}",
            True,
            BLACK
        )

        self.screen.blit(
            text,
            (10, 10)
        )