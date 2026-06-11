import pygame

from environment.game import ForagingGame
from visualization.pygame_renderer import Renderer


def main():

    env = ForagingGame()

    renderer = Renderer(env)

    clock = pygame.time.Clock()

    running = True

    while running:

        action = None

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:
                    action = 0

                elif event.key == pygame.K_DOWN:
                    action = 1

                elif event.key == pygame.K_LEFT:
                    action = 2

                elif event.key == pygame.K_RIGHT:
                    action = 3

                elif event.key == pygame.K_SPACE:
                    action = 4

        if action is not None:

            state, reward, done = env.step(action)

            print(
                f"reward={reward:.2f}"
            )

            if done:

                print(
                    f"Episode finished. "
                    f"Tokens collected = "
                    f"{env.player.tokens}"
                )

                env.reset()

        renderer.draw()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()