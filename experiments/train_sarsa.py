from environment import ForagingGame
from agents import SARSAAgent
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sarsa_agent.pkl"

EPISODES = 50000


def main():

    env = ForagingGame(
        threat_probability=0.5,
        realtime=False,
        steps_per_second=10,
    )

    agent = SARSAAgent(
        alpha=0.15,
        gamma=0.95,
        epsilon=0.9
    )

    episode_rewards = []

    for episode in range(EPISODES):

        state = env.reset()

        action = agent.choose_action(state)

        total_reward = 0

        done = False

        while not done:

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            total_reward += reward

            if done:

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    action,
                    True
                )

                break

            next_action = agent.choose_action(
                next_state
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                False
            )

            state = next_state
            action = next_action

        agent.decay_epsilon()

        episode_rewards.append(
            total_reward
        )

        if episode % 100 == 0:

            avg_reward = sum(
                episode_rewards[-100:]
            ) / max(
                1,
                len(episode_rewards[-100:])
            )

            print(
                f"Episode {episode} | "
                f"Avg Reward {avg_reward:.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

    agent.save(MODEL_PATH)
    print("Saved trained agent to sarsa_agent.pkl")
    print("Training finished.")


if __name__ == "__main__":
    main()