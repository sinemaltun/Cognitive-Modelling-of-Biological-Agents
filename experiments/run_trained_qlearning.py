import time
from pathlib import Path

import pygame

from environment import ForagingGame
from agents import QLearningAgent
from visualization import PygameRenderer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "qlearning_agent.pkl"


def main():
    env = ForagingGame(
        threat_probability=0.5,
        realtime=True,
    )

    renderer = PygameRenderer(env)

    agent = QLearningAgent(epsilon=0.0)
    agent.load(MODEL_PATH)

    print(f"Loaded trained Q-learning agent from {MODEL_PATH}")

    state = env.reset()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.greedy_action(state)

        next_state, reward, done, info = env.step(action)

        renderer.draw()

        state = next_state

        clock.tick(10)

        if done:
            print(
                f"Episode finished | "
                f"Status: {info['status'].name} | "
                f"Tokens: {info['tokens_collected']}"
            )

            time.sleep(1)
            state = env.reset()

    renderer.close()


if __name__ == "__main__":
    main()