import time

from environment import ForagingGame
from agents import RandomAgent
from visualization import PygameRenderer


def main():

    env = ForagingGame(
        threat_probability=0.5
    )

    agent = RandomAgent()

    renderer = PygameRenderer(env)

    state = env.reset()

    while True:

        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        renderer.draw()

        state = next_state

        time.sleep(0.15)

        if done:

            print(
                f"Status={info['status'].name}"
                f" Tokens={info['tokens_collected']}"
            )

            state = env.reset()


if __name__ == "__main__":
    main()