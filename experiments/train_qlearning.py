from pathlib import Path

from environment import ForagingGame
from agents import QLearningAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "qlearning_agent.pkl"

EPISODES = 100000


def main():
    env = ForagingGame(
        threat_probability=0.7,
        realtime=False,
        steps_per_second=10,
    )

    agent = QLearningAgent(
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon=0.05,
    )

    episode_rewards = []

    for episode in range(EPISODES):
        state = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            total_reward += reward
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

            print(
                f"Episode {episode} | "
                f"Avg Reward {avg_reward:.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

    agent.save(MODEL_PATH)

    print(f"Saved trained Q-learning agent to {MODEL_PATH}")
    print("Training finished.")


if __name__ == "__main__":
    main()