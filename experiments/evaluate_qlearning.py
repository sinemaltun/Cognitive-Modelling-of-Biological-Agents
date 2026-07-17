import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.qlearning_agent import (
    QLearningAgent,
)

from environment import ForagingGame

from evaluation import (
    CSVLogger,
    EpisodeTracker,
    RunStatistics,
    save_run_config,
    save_run_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_EPISODES = 240
BASE_SEED = 1_000


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-learning agent without rendering.")

    parser.add_argument(
        "model_path",
        type=Path,
        help=(
            "Path to the trained Q-learning "
            "pickle file."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=EVALUATION_EPISODES,
    )

    parser.add_argument(
        "--threat-probability",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = args.model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: {model_path}"
        )

    run_id = ("qlearning_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    run_dir = (PROJECT_ROOT / "results" / run_id)

    env = ForagingGame(
        threat_probability=args.threat_probability,
        realtime=False,
        steps_per_second=10,
        action_noise=args.action_noise,
    )

    agent = QLearningAgent(epsilon=0.0)
    agent.load(model_path)

    logger = CSVLogger(run_dir)

    save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "mode": "evaluation",
            "model_type": "qlearning",
            "model_path": str(model_path),
            "episodes": args.episodes,
            "base_seed": args.base_seed,

            "environment": {
                "threat_probability": env.threat_probability,
                "action_noise": env.action_noise,
                "steps_per_second": env.steps_per_second,
                "trial_duration": env.trial_duration,
                "chase_duration": env.chase_duration,
                "rewards": env.rewards,
            },

            "agent": agent.metadata(),
        },
    )

    run_statistics = RunStatistics()

    for episode in range(args.episodes):
        episode_seed = (args.base_seed + episode)

        random.seed(episode_seed)
        np.random.seed(episode_seed)

        state = env.reset()

        tracker = EpisodeTracker(
            run_id=run_id,
            model_type="qlearning",
            mode="evaluation",
            episode=episode,
            run_seed=args.base_seed,
            episode_seed=episode_seed,
        )

        tracker.start(env)

        done = False
        info = None

        while not done:
            action = agent.greedy_action(state)

            (next_state, reward, done, info,) = env.step(action)

            logger.log_step(
                tracker.record_step(
                    env=env,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            state = next_state

        episode_record = tracker.finish(
            env=env,
            info=info,
            epsilon_used=0.0,
            q_table_states=agent.q_table_size,
        )

        logger.log_episode(episode_record)

        run_statistics.update(episode_record)

    save_run_summary(
        run_dir,
        {
            "run_id": run_id,
            "mode": "evaluation",
            "model_type": "qlearning",
            "model_path": str(model_path),
            "base_seed": args.base_seed,
            "final_agent": agent.metadata(),
            **run_statistics.to_summary_dict(),
        },
    )

    print(f"Evaluation saved to {run_dir}")


if __name__ == "__main__":
    main()