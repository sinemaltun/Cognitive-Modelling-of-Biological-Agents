import random
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.sarsa_agent import SARSAAgent

from environment import ForagingGame
from evaluation import (
    CSVLogger,
    EpisodeTracker,
    RunStatistics,
    TrainingProgressRecord,
    save_run_config,
    save_run_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True,)

RESULTS_DIR.mkdir(parents=True, exist_ok=True,)

RUN_ID = ("sarsa_training_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

RUN_DIR = RESULTS_DIR / RUN_ID
MODEL_PATH = MODEL_DIR / f"{RUN_ID}.pkl"

EPISODES = 50_000
SEED = 42

PROGRESS_WINDOW = 100
LOG_STEP_EVERY_N_EPISODES = 500


def mean_field(records: list, field_name: str,) -> float:
    if not records:
        return 0.0

    return sum(getattr(record, field_name) for record in records) / len(records)


def calculate_threat_survival_rate(records: list,) -> tuple[int, float | None]:
    threat_records = [record for record in records if record.threat_trial]

    if not threat_records:
        return 0, None

    survived_count = sum(record.survived_threat is True for record in threat_records)

    return (len(threat_records), survived_count / len(threat_records),)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    env = ForagingGame(
        threat_probability=0.5,
        realtime=False,
        steps_per_second=10,
        action_noise=0.2,
    )

    agent = SARSAAgent(
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
    )

    logger = CSVLogger(RUN_DIR)

    save_run_config(
        RUN_DIR,
        {
            "run_id": RUN_ID,
            "mode": "training",
            "model_type": "sarsa",
            "run_seed": SEED,
            "episodes": EPISODES,
            "model_path": str(MODEL_PATH),

            "logging": {
                "progress_window": PROGRESS_WINDOW,
                "log_step_every_n_episodes": LOG_STEP_EVERY_N_EPISODES,
            },

            "environment": {
                "width": env.grid.width,
                "height": env.grid.height,

                "threat_probability": env.threat_probability,

                "trial_duration": env.trial_duration,

                "chase_duration": env.chase_duration,

                "steps_per_second": env.steps_per_second,

                "action_noise": env.action_noise,

                "rewards": env.rewards,
            },

            "initial_agent": agent.metadata(),
        },
    )

    recent_records = []
    whole_run_statistics = RunStatistics()

    print(f"Starting run: {RUN_ID}")
    print(f"Results directory: {RUN_DIR}")

    for episode in range(EPISODES):
        epsilon_used = agent.epsilon

        state = env.reset()
        action = agent.choose_action(state)

        tracker = EpisodeTracker(
            run_id=RUN_ID,
            model_type="sarsa",
            mode="training",
            episode=episode,
            run_seed=SEED,
            episode_seed=None,
        )

        tracker.start(env)

        done = False
        info = None

        while not done:
            (next_state, reward, done, info,) = env.step(action)

            step_record = tracker.record_step(
                env=env,
                reward=reward,
                done=done,
                info=info,
            )

            if (episode % LOG_STEP_EVERY_N_EPISODES == 0):
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

            next_action = agent.choose_action(next_state)

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

        if info is None:
            raise RuntimeError("The episode ended without environment information.")

        episode_record = tracker.finish(
            env=env,
            info=info,
            epsilon_used=epsilon_used,
            q_table_states=agent.q_table_size,
        )

        logger.log_episode(episode_record)

        whole_run_statistics.update(episode_record)

        recent_records.append(episode_record)

        if len(recent_records) > PROGRESS_WINDOW:
            recent_records.pop(0)

        agent.decay_epsilon()
        epsilon_next = agent.epsilon

        should_log_progress = ((episode + 1) % PROGRESS_WINDOW == 0 or episode == EPISODES - 1)

        if (should_log_progress and recent_records):
            (threat_trial_count, threat_survival_rate,) = calculate_threat_survival_rate(recent_records)

            progress = TrainingProgressRecord(
                run_id=RUN_ID,
                model_type="sarsa",

                episode=episode,
                window_size=len(recent_records),

                mean_reward=mean_field(recent_records,"total_reward",),

                mean_tokens_gross=mean_field(recent_records,"tokens_collected_gross",),

                mean_tokens_retained=mean_field(recent_records,"tokens_retained",),

                survival_rate=mean_field(recent_records,"survived",),

                threat_trial_count=(threat_trial_count),

                threat_survival_rate=(threat_survival_rate),

                caught_rate=mean_field(recent_records,"caught",),

                escape_rate=mean_field(recent_records,"escaped",),

                epsilon_used=epsilon_used,
                epsilon_next=epsilon_next,

                q_table_states=agent.q_table_size,
            )

            logger.log_training_progress(progress)

            threat_text = (
                f"{threat_survival_rate:.2%}"
                if threat_survival_rate
                is not None
                else "n/a"
            )

            print(
                f"Episode "
                f"{episode + 1}/{EPISODES} | "
                f"Average reward "
                f"{progress.mean_reward:.2f} | "
                f"Overall survival "
                f"{progress.survival_rate:.2%} | "
                f"Threat survival "
                f"{threat_text} | "
                f"Epsilon used "
                f"{epsilon_used:.4f} | "
                f"Next epsilon "
                f"{epsilon_next:.4f} | "
                f"Q states "
                f"{agent.q_table_size}"
            )

    agent.save(MODEL_PATH)

    (final_threat_trial_count, final_threat_survival_rate,) = calculate_threat_survival_rate(recent_records)

    save_run_summary(
        RUN_DIR,
        {
            "run_id": RUN_ID,
            "mode": "training",
            "model_type": "sarsa",
            "run_seed": SEED,
            "episodes_completed": EPISODES,
            "model_path": str(MODEL_PATH),

            "final_agent": agent.metadata(),

            "whole_run": whole_run_statistics.to_summary_dict(),

            "final_window": {
                "window_size": len(recent_records),

                "mean_reward": (
                    mean_field(recent_records,"total_reward",)
                    if recent_records
                    else None
                ),

                "mean_tokens_gross": (
                    mean_field(recent_records,"tokens_collected_gross",)
                    if recent_records
                    else None
                ),

                "mean_tokens_retained": (
                    mean_field(recent_records,"tokens_retained",)
                    if recent_records
                    else None
                ),

                "overall_survival_rate": (
                    mean_field(recent_records,"survived",)
                    if recent_records
                    else None
                ),

                "caught_rate": (mean_field(recent_records,"caught",)
                    if recent_records
                    else None
                ),

                "escape_rate": (mean_field(recent_records,"escaped",)
                    if recent_records
                    else None
                ),

                "threat_trial_count": final_threat_trial_count,

                "threat_survival_rate": final_threat_survival_rate,
            },
        },
    )

    print("Saved trained SARSA agent to "f"{MODEL_PATH}")

    print(f"Logged results to {RUN_DIR}")

    print("Training finished.")


if __name__ == "__main__":
    main()