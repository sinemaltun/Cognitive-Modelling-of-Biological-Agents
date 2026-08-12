import pygame

from agents.base_agent import BaseAgent
from environment import Action


class HumanAgent(BaseAgent):
    """
    Human-controlled agent.

    This class is responsible only for reading keyboard input.

    It does NOT control movement timing. The experiment loop in
    run_human.py determines when environment steps occur.
    """

    def __init__(self):
        self.quit_requested = False
        self.current_action = None

    def update_input(self) -> None:
        """
        Process Pygame events and determine which movement key
        is currently being held.

        This method should be called once per rendered frame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.quit_requested = True
                return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.current_action = Action.UP

        elif keys[pygame.K_RIGHT]:
            self.current_action = Action.RIGHT

        elif keys[pygame.K_DOWN]:
            self.current_action = Action.DOWN

        elif keys[pygame.K_LEFT]:
            self.current_action = Action.LEFT

        elif keys[pygame.K_SPACE]:
            self.current_action = Action.STAY

        else:
            self.current_action = None

    def choose_action(self, state=None):
        """
        Return the currently held movement action.

        None means that no movement key is currently pressed.
        run_human.py converts this to Action.STAY when the next
        environment step occurs.
        """
        return self.current_action