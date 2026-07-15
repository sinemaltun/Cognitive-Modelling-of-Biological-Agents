import pygame
CELL_SIZE = 32
GRID_LINE_WIDTH = 1
FRAME_WIDTH = 8
INFO_PANEL_HEIGHT = 48

WHITE, LIGHT_GRAY, DARK_GRAY = (255, 255, 255), (220, 220, 220), (120, 120, 120)
BLACK, GREEN, YELLOW, RED, BLUE = (0, 0, 0), (0, 170, 0), (255, 220, 0), (220, 0, 0), (0, 80, 220)

class PygameRenderer:
    def __init__(self, env, cell_size=CELL_SIZE):
        pygame.init()
        self.env = env
        self.cell_size = cell_size
        self.grid_width_px = env.width * cell_size
        self.grid_height_px = env.height * cell_size
        self.screen_width = self.grid_width_px
        self.screen_height = self.grid_height_px + INFO_PANEL_HEIGHT
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Foraging Threat Task")
        
        self.font = pygame.font.SysFont(None, 26)
        self.small_font = pygame.font.SysFont(None, 22)

    def draw(self, is_chase=False, status="SAFE"):
        self.screen.fill(WHITE)
        self._draw_grid()
        self._draw_safe_zone()
        self._draw_tokens()
        self._draw_predator(is_chase)
        self._draw_player()
        self._draw_frame(is_chase)
        self._draw_info_panel(is_chase, status)
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def _cell_rect(self, entity):
        return pygame.Rect(entity.x * self.cell_size, entity.y * self.cell_size, self.cell_size, self.cell_size)

    def _cell_center(self, entity):
        return (entity.x * self.cell_size + self.cell_size // 2, entity.y * self.cell_size + self.cell_size // 2)

    def _draw_grid(self):
        for x in range(self.env.width):
            for y in range(self.env.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, LIGHT_GRAY, rect, GRID_LINE_WIDTH)

    def _draw_player(self):
        x = self.env.player.x * self.cell_size
        y = self.env.player.y * self.cell_size
        triangle = [(x + self.cell_size // 2, y + 4), (x + 5, y + self.cell_size - 5), (x + self.cell_size - 5, y + self.cell_size - 5)]
        pygame.draw.polygon(self.screen, GREEN, triangle)

    def _draw_tokens(self):
        radius = max(6, self.cell_size // 4)
        for token in self.env.tokens:
            cx, cy = self._cell_center(token)
            diamond = [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]
            pygame.draw.polygon(self.screen, YELLOW, diamond)
            pygame.draw.polygon(self.screen, BLACK, diamond, 1)

    def _draw_predator(self, is_chase):
        color = RED if is_chase else DARK_GRAY
        pygame.draw.circle(self.screen, color, self._cell_center(self.env.predator), self.cell_size // 3)

    def _draw_safe_zone(self):
        pygame.draw.rect(self.screen, BLACK, self._cell_rect(self.env.safe_zone))

    def _draw_frame(self, is_chase):
        color = RED if is_chase else BLUE
        frame_rect = pygame.Rect(0, 0, self.grid_width_px, self.grid_height_px)
        pygame.draw.rect(self.screen, color, frame_rect, FRAME_WIDTH)

    def _draw_info_panel(self, is_chase, status):
        panel_y = self.grid_height_px
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, panel_y, self.screen_width, INFO_PANEL_HEIGHT))
        pygame.draw.line(self.screen, BLACK, (0, panel_y), (self.screen_width, panel_y), 2)

        phase_text = "CHASE" if is_chase else "FORAGE"
        tokens_text = f"Tokens: {self.env.player.tokens}"
        time_text = f"Steps: {self.env.step_counter}"

        left_text = self.font.render(tokens_text, True, BLACK)
        middle_text = self.small_font.render(f"Phase: {phase_text} | Status: {status}", True, BLACK)
        right_text = self.font.render(time_text, True, BLACK)

        self.screen.blit(left_text, (12, panel_y + 13))
        self.screen.blit(middle_text, (self.screen_width // 2 - middle_text.get_width() // 2, panel_y + 15))
        self.screen.blit(right_text, (self.screen_width - right_text.get_width() - 12, panel_y + 13))