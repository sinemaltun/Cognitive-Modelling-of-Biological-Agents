import argparse
import random
from datetime import datetime
from pathlib import Path

import pygame

from agents.human_agent import HumanAgent
from environment import ForagingGame

from evaluation import (
    CSVLogger,
    EpisodeTracker,
    RunStatistics,
    save_run_config,
    save_run_summary,
)

from visualization import PygameRenderer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run and log human participant trials.")

    parser.add_argument(
        "--episodes",
        type=int,
        default=240,
    )

    parser.add_argument(
        "--participant-id",
        type=str,
        default="anonymous",
    )

    parser.add_argument(
        "--threat-probabilities",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help=(
            "Threat probabilities sampled uniformly "
            "across evaluation episodes."
        ),
    )

    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    run_id = (
        f"human_{args.participant_id}_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_dir = (
        PROJECT_ROOT
        / "results"
        / run_id
    )

    # Initial value only.
    # A new threat probability is selected
    # before every episode reset.
    env = ForagingGame(
        threat_probability=args.threat_probabilities[0],
        realtime=True,
        action_noise=args.action_noise,
    )

    agent = HumanAgent()
    renderer = PygameRenderer(env)
    clock = pygame.time.Clock()

    logger = CSVLogger(run_dir)

    save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "mode": "human",
            "model_type": "human",
            "participant_id": args.participant_id,
            "episodes": args.episodes,

            "environment": {
                "threat_probabilities": args.threat_probabilities,

                "threat_sampling": "uniform_discrete",

                "action_noise": env.action_noise,

                "trial_duration": env.trial_duration,

                "chase_duration": env.chase_duration,

                "realtime": True,
                "rewards": env.rewards,
            },
        },
    )

    # Select the threat condition for the first episode.
    env.threat_probability = random.choice(args.threat_probabilities)

    env.reset()

    episode = 0

    tracker = EpisodeTracker(
        run_id=run_id,
        model_type="human",
        mode="human",
        episode=episode,
        run_seed=None,
        episode_seed=None,
    )

    tracker.start(env)

    run_statistics = RunStatistics()

    running = True

    while running and episode < args.episodes:
        action = agent.choose_action()

        if agent.quit_requested:
            running = False
            continue

        if action is not None:
            (
                _,
                reward,
                done,
                info,
            ) = env.step(action)

            logger.log_step(
                tracker.record_step(
                    env=env,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            if done:
                episode_record = tracker.finish(env=env,info=info,)

                logger.log_episode(episode_record)

                run_statistics.update(episode_record)

                print(
                    f"Episode {episode} finished | "
                    f"Threat probability: "
                    f"{env.threat_probability:.1f} | "
                    f"Status: "
                    f"{episode_record.status} | "
                    f"Gross tokens: "
                    f"{episode_record.tokens_collected_gross}"
                )

                episode += 1

                if episode >= args.episodes:
                    running = False

                else:
                    # IMPORTANT:
                    # Choose the next threat condition
                    # before calling reset().
                    env.threat_probability = random.choice(args.threat_probabilities)

                    env.reset()

                    tracker = EpisodeTracker(
                        run_id=run_id,
                        model_type="human",
                        mode="human",
                        episode=episode,
                        run_seed=None,
                        episode_seed=None,
                    )

                    tracker.start(env)

        renderer.draw()
        clock.tick(60)

    renderer.close()

    save_run_summary(
        run_dir,
        {
            "run_id": run_id,
            "mode": "human",
            "model_type": "human",
            "participant_id": args.participant_id,

            "threat_probabilities": args.threat_probabilities,

            "threat_sampling": "uniform_discrete",

            **run_statistics.to_summary_dict(),
        },
    )

    print(f"Human results saved to {run_dir}")


if __name__ == "__main__":
    main()