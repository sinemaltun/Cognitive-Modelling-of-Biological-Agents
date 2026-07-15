import time
from pathlib import Path

import pygame

from agents import (
    PhaseAwareValueIterationAgent,
    ValueIterationAgent,
)

from environment import ForagingGame
from visualization import PygameRenderer


PROJECT_ROOT = (Path(__file__).resolve().parent.parent)

MODEL_DIR = PROJECT_ROOT / "models"

SERIAL = 1

FORAGE_MODEL_PATH = (MODEL_DIR / f"value_iteration_forage_{SERIAL:03d}.pkl")

CHASE_MODEL_PATH = (MODEL_DIR / f"value_iteration_chase_{SERIAL:03d}.pkl")


def main():
    env = ForagingGame(
        threat_probability=0.7,
        realtime=True,
    )

    forage_agent = ValueIterationAgent()
    chase_agent = ValueIterationAgent()

    forage_agent.load(FORAGE_MODEL_PATH)
    chase_agent.load(CHASE_MODEL_PATH)

    agent = PhaseAwareValueIterationAgent(
        forage_agent=forage_agent,
        chase_agent=chase_agent,
    )

    renderer = PygameRenderer(env)
    clock = pygame.time.Clock()

    env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.choose_action(env)

        _, _, done, info = env.step(action)

        renderer.draw()
        clock.tick(10)

        if done:
            print(
                "Episode finished | "
                f"Status: {info['status'].name} | "
                f"Tokens: "
                f"{info['tokens_collected']}"
            )

            time.sleep(1)
            env.reset()

    renderer.close()


if __name__ == "__main__":
    main()