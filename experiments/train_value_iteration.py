from datetime import datetime
from pathlib import Path

from agents import ValueIterationAgent
from evaluation import (
    CSVLogger,
    ValueIterationProgressRecord,
    save_run_config,
)
from planning import GridPlanningModel


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
    "value_iteration_training_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

RUN_DIR = RESULTS_DIR / RUN_ID

FORAGE_MODEL_PATH = (
    MODEL_DIR / f"{RUN_ID}_forage.pkl"
)

CHASE_MODEL_PATH = (
    MODEL_DIR / f"{RUN_ID}_chase.pkl"
)


def log_training_history(
    logger: CSVLogger,
    agent: ValueIterationAgent,
    phase: str,
) -> None:
    """
    Write one CSV row for every Bellman sweep performed by
    a Value Iteration agent.
    """
    for row in agent.training_history:
        logger.log_value_iteration_progress(
            ValueIterationProgressRecord(
                run_id=RUN_ID,
                phase=phase,
                iteration=row["iteration"],
                maximum_value_change=row[
                    "maximum_value_change"
                ],
                state_count=row["state_count"],
                converged=row["converged"],
            )
        )


def main():
    model = GridPlanningModel(
        width=24,
        height=16,
        predator_speed=2,
        action_noise=0.0,
        caught_penalty=-100.0,
    )

    forage_agent = ValueIterationAgent(
        gamma=0.95,
        theta=1e-5,
        max_iterations=500,
    )

    chase_agent = ValueIterationAgent(
        gamma=0.95,
        theta=1e-5,
        max_iterations=500,
    )

    logger = CSVLogger(RUN_DIR)

    save_run_config(
        RUN_DIR,
        {
            "run_id": RUN_ID,
            "mode": "training",
            "model_type": "value_iteration",

            "model_paths": {
                "foraging": str(
                    FORAGE_MODEL_PATH
                ),
                "chase": str(
                    CHASE_MODEL_PATH
                ),
            },

            "planning_model": {
                "width": model.width,
                "height": model.height,
                "safe_x": model.safe_x,
                "safe_y": model.safe_y,
                "predator_speed": (
                    model.predator_speed
                ),
                "action_noise": (
                    model.action_noise
                ),
                "caught_penalty": (
                    model.caught_penalty
                ),
                "rewards": model.rewards,
            },

            "forage_agent": (
                forage_agent.metadata()
            ),

            "chase_agent": (
                chase_agent.metadata()
            ),
        },
    )

    print(f"Starting run: {RUN_ID}")
    print(f"Results directory: {RUN_DIR}")

    print("Training foraging policy...")

    forage_agent.train(
        states=model.generate_forage_states(),
        transition_function=(
            model.forage_transitions
        ),
    )

    log_training_history(
        logger=logger,
        agent=forage_agent,
        phase="foraging",
    )

    print("Training chase policy...")

    chase_agent.train(
        states=model.generate_chase_states(),
        transition_function=(
            model.chase_transitions
        ),
    )

    log_training_history(
        logger=logger,
        agent=chase_agent,
        phase="chase",
    )

    forage_agent.save(
        FORAGE_MODEL_PATH
    )

    chase_agent.save(
        CHASE_MODEL_PATH
    )

    print(
        "Saved foraging policy to "
        f"{FORAGE_MODEL_PATH}"
    )

    print(
        "Saved chase policy to "
        f"{CHASE_MODEL_PATH}"
    )

    print(
        "Saved convergence history to "
        f"{RUN_DIR / 'value_iteration_progress.csv'}"
    )

    print("Training finished.")


if __name__ == "__main__":
    main()