import random
from datetime import datetime
from pathlib import Path

import numpy as np

from agents import SARSAAgent
from environment import ForagingGame

from evaluation import (
    CSVLogger,
    EpisodeTracker,
    save_run_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TODO
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "REPLACE_WITH_SARSA_MODEL.pkl"
)

RUN_ID = (
    "sarsa_evaluation_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

RUN_DIR = PROJECT_ROOT / "results" / RUN_ID

EVALUATION_EPISODES = 240
BASE_SEED = 1_000


def main():
    env = ForagingGame(
        threat_probability=0.2,
        realtime=False,
        steps_per_second=10,
        action_noise=0.0,
    )

    agent = SARSAAgent(epsilon=0.0)
    agent.load(MODEL_PATH)

    logger = CSVLogger(RUN_DIR)

    save_run_config(
        RUN_DIR,
        {
            "run_id": RUN_ID,
            "mode": "evaluation",
            "model_type": "sarsa",
            "model_path": str(MODEL_PATH),
            "episodes": EVALUATION_EPISODES,
            "base_seed": BASE_SEED,
            "environment": {
                "threat_probability": (
                    env.threat_probability
                ),
                "action_noise": env.action_noise,
                "steps_per_second": (
                    env.steps_per_second
                ),
                "rewards": env.rewards,
            },
            "agent": agent.metadata(),
        },
    )

    for episode in range(EVALUATION_EPISODES):
        episode_seed = BASE_SEED + episode

        random.seed(episode_seed)
        np.random.seed(episode_seed)

        state = env.reset()

        tracker = EpisodeTracker(
            run_id=RUN_ID,
            model_type="sarsa",
            mode="evaluation",
            episode=episode,
            seed=episode_seed,
        )

        tracker.start(env)

        done = False
        info = None

        while not done:
            action = agent.greedy_action(state)

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            logger.log_step(
                tracker.record_step(
                    env,
                    reward,
                    done,
                    info,
                )
            )

            state = next_state

        logger.log_episode(
            tracker.finish(
                env,
                info,
                epsilon=0.0,
                q_table_states=agent.q_table_size,
            )
        )

    print(f"Evaluation saved to {RUN_DIR}")


if __name__ == "__main__":
    main()