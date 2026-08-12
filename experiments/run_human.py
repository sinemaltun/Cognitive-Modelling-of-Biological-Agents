import argparse
import random
from datetime import datetime
from pathlib import Path

import pygame

from agents.human_agent import HumanAgent

from environment import (
    Action,
    ForagingGame,
)

from evaluation import (
    CSVLogger,
    EpisodeTracker,
    RunStatistics,
    save_run_config,
    save_run_summary,
)

from visualization import PygameRenderer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# The game/environment advances at 10 steps per second.
GAME_STEPS_PER_SECOND = 10

# 10 steps per second = one step every 100 ms.
STEP_INTERVAL_MS = (
    1000 // GAME_STEPS_PER_SECOND
)

# Rendering is independent from game logic.
RENDER_FPS = 60


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
        default=[0.8],
        help=(
            "Threat probabilities sampled uniformly "
            "across human experiment episodes."
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

    # Validate threat probabilities.
    for probability in args.threat_probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("All threat probabilities must ""be between 0.0 and 1.0.")

    if not 0.0 <= args.action_noise <= 1.0:
        raise ValueError("Action noise must be between ""0.0 and 1.0.")

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

    # Initial threat probability only.
    # A new value is selected before every episode.
    env = ForagingGame(
        threat_probability=args.threat_probabilities[0],
        realtime=True,
        steps_per_second=GAME_STEPS_PER_SECOND,
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

                "steps_per_second": GAME_STEPS_PER_SECOND,

                "render_fps": RENDER_FPS,

                "trial_duration": env.trial_duration,

                "chase_duration": env.chase_duration,

                "realtime": True,

                "rewards": env.rewards,
            },
        },
    )

    # ---------------------------------------------------------
    # First episode
    # ---------------------------------------------------------

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

    # Time of the previous environment step.
    last_step_time = pygame.time.get_ticks()

    running = True

    # ---------------------------------------------------------
    # Main experiment loop
    # ---------------------------------------------------------

    while running and episode < args.episodes:
        # Read keyboard input every rendered frame.
        agent.update_input()

        if agent.quit_requested:
            running = False
            continue

        current_time = pygame.time.get_ticks()

        # -----------------------------------------------------
        # Environment clock
        #
        # The screen may render at 60 FPS, but the game only advances once every STEP_INTERVAL_MS.
        # -----------------------------------------------------

        if current_time - last_step_time >= STEP_INTERVAL_MS:
            last_step_time = current_time

            action = agent.choose_action()

            # No key pressed means the participant chooses not to move during this game step.
            # Importantly, the environment still advances.
            if action is None:
                action = Action.STAY

            (
                _,
                reward,
                done,
                info,
            ) = env.step(action)

            step_record = (
                tracker.record_step(
                    env=env,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )

            logger.log_step(step_record)

            # -------------------------------------------------
            # Episode finished
            # -------------------------------------------------

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

                # ---------------------------------------------
                # All requested episodes completed
                # ---------------------------------------------

                if episode >= args.episodes:
                    running = False

                # ---------------------------------------------
                # Start next episode
                # ---------------------------------------------

                else:
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

                    # Start the new episode's environment
                    # clock from the current time.
                    last_step_time = pygame.time.get_ticks()

        # -----------------------------------------------------
        # Rendering
        #
        # Rendering remains much faster than environment
        # stepping so the UI stays responsive.
        # -----------------------------------------------------

        renderer.draw()

        clock.tick(RENDER_FPS)

    # ---------------------------------------------------------
    # Shutdown / summary
    # ---------------------------------------------------------

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

            "game_steps_per_second": GAME_STEPS_PER_SECOND,

            **run_statistics.to_summary_dict(),
        },
    )

    print(f"Human results saved to {run_dir}")


if __name__ == "__main__":
    main()