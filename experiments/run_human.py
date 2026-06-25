import pygame

from environment import ForagingGame
from agents import HumanAgent
from visualization import PygameRenderer


def main():

    env = ForagingGame(
        threat_probability=0.2,
        realtime = True,
    )

    agent = HumanAgent()

    renderer = PygameRenderer(env)

    clock = pygame.time.Clock()

    state = env.reset()

    running = True

    while running:

        action = agent.choose_action(state)

        if agent.quit_requested:
            break

        if action is not None:

            next_state, reward, done, info = env.step(action)

            state = next_state

            if done:

                print(
                    f"Episode finished. "
                    f"Status={info['status'].name} "
                    f"Tokens={info['tokens_collected']}"
                )

                state = env.reset()

        renderer.draw()

        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()