import random
from datetime import datetime
from pathlib import Path

import numpy as np

from agents import SARSAAgent
from environment import ForagingGame

from evaluation import (
    CSVLogger,
    EpisodeTracker,
    TrainingProgressRecord,
    save_run_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_ID = (
    "sarsa_training_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

RUN_DIR = RESULTS_DIR / RUN_ID
RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / f"{RUN_ID}.pkl"

EPISODES = 50_000

SEED = 42
PROGRESS_WINDOW = 100
LOG_STEP_EVERY_N_EPISODES = 500


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    env = ForagingGame(
        threat_probability=0.8,
        realtime=False,
        steps_per_second=10,
        action_noise=0.0,
    )

    agent = SARSAAgent(
        alpha=0.15,
        gamma=0.95,
        epsilon=0.9,
    )

    logger = CSVLogger(RUN_DIR)

    save_run_config(
        RUN_DIR,
        {
            "run_id": RUN_ID,
            "mode": "training",
            "model_type": "sarsa",
            "seed": SEED,
            "episodes": EPISODES,
            "model_path": str(MODEL_PATH),

            "logging": {
                "progress_window": PROGRESS_WINDOW,
                "log_step_every_n_episodes": (
                    LOG_STEP_EVERY_N_EPISODES
                ),
            },

            "environment": {
                "width": env.grid.width,
                "height": env.grid.height,
                "threat_probability": (
                    env.threat_probability
                ),
                "trial_duration": (
                    env.trial_duration
                ),
                "chase_duration": (
                    env.chase_duration
                ),
                "steps_per_second": (
                    env.steps_per_second
                ),
                "action_noise": (
                    env.action_noise
                ),
                "rewards": env.rewards,
            },

            "agent": agent.metadata(),
        },
    )

    recent_records = []

    print(f"Starting run: {RUN_ID}")
    print(f"Results directory: {RUN_DIR}")

    for episode in range(EPISODES):
        state = env.reset()
        action = agent.choose_action(state)

        tracker = EpisodeTracker(
            run_id=RUN_ID,
            model_type="sarsa",
            mode="training",
            episode=episode,
            seed=SEED,
        )

        tracker.start(env)

        done = False
        info = env._info()

        while not done:
            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            step_record = tracker.record_step(
                env=env,
                reward=reward,
                done=done,
                info=info,
            )

            if (
                episode
                % LOG_STEP_EVERY_N_EPISODES
                == 0
            ):
                logger.log_step(step_record)

            if done:
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=action,
                    done=True,
                )

                break

            next_action = agent.choose_action(
                next_state
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=False,
            )

            state = next_state
            action = next_action

        episode_record = tracker.finish(
            env=env,
            info=info,
            epsilon=agent.epsilon,
            q_table_states=agent.q_table_size,
        )

        logger.log_episode(episode_record)

        recent_records.append(episode_record)

        if len(recent_records) > PROGRESS_WINDOW:
            recent_records.pop(0)

        agent.decay_epsilon()

        if (
            episode % PROGRESS_WINDOW == 0
            and recent_records
        ):
            window = recent_records

            progress = TrainingProgressRecord(
                run_id=RUN_ID,
                model_type="sarsa",

                episode=episode,
                window_size=len(window),

                mean_reward=sum(
                    row.total_reward
                    for row in window
                ) / len(window),

                mean_tokens_gross=sum(
                    row.tokens_collected_gross
                    for row in window
                ) / len(window),

                mean_tokens_retained=sum(
                    row.tokens_retained
                    for row in window
                ) / len(window),

                survival_rate=sum(
                    row.survived
                    for row in window
                ) / len(window),

                caught_rate=sum(
                    row.caught
                    for row in window
                ) / len(window),

                escape_rate=sum(
                    row.escaped
                    for row in window
                ) / len(window),

                epsilon=agent.epsilon,
                q_table_states=agent.q_table_size,
            )

            logger.log_training_progress(
                progress
            )

            print(
                f"Episode {episode} | "
                f"Avg Reward "
                f"{progress.mean_reward:.2f} | "
                f"Survival "
                f"{progress.survival_rate:.2%} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

    agent.save(MODEL_PATH)

    print(f"Saved trained SARSA agent to {MODEL_PATH}")
    print(f"Logged results to {RUN_DIR}")
    print("Training finished.")


if __name__ == "__main__":
    main()