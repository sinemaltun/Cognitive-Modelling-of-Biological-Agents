import pygame

from agents.base_agent import BaseAgent
from environment import Action


class HumanAgent(BaseAgent):
    def __init__(self):
        self.last_action = None
        self.quit_requested = False

    def choose_action(self, state=None):
        self.last_action = None

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.quit_requested = True
                return None

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:
                    self.last_action = Action.UP

                elif event.key == pygame.K_RIGHT:
                    self.last_action = Action.RIGHT

                elif event.key == pygame.K_DOWN:
                    self.last_action = Action.DOWN

                elif event.key == pygame.K_LEFT:
                    self.last_action = Action.LEFT

                elif event.key == pygame.K_SPACE:
                    self.last_action = Action.STAY

                elif event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                    return None

        return self.last_action