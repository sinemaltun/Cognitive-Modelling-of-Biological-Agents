import pygame

from agents.base_agent import BaseAgent
from environment import Action


class HumanAgent(BaseAgent):
    def __init__(self, movement_interval_ms: int = 120):
        if movement_interval_ms <= 0:
            raise ValueError(
                "movement_interval_ms must be greater than zero."
            )

        self.movement_interval_ms = movement_interval_ms
        self.next_movement_time = 0
        self.quit_requested = False

    @staticmethod
    def _get_pressed_action():
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return Action.UP

        if keys[pygame.K_RIGHT]:
            return Action.RIGHT

        if keys[pygame.K_DOWN]:
            return Action.DOWN

        if keys[pygame.K_LEFT]:
            return Action.LEFT

        if keys[pygame.K_SPACE]:
            return Action.STAY

        return None

    def choose_action(self, state=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return None

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.quit_requested = True
                return None

        action = self._get_pressed_action()

        if action is None:
            return None

        current_time = pygame.time.get_ticks()

        if current_time < self.next_movement_time:
            return None

        self.next_movement_time = (
            current_time + self.movement_interval_ms
        )

        return action