from pathlib import Path

from agents import ValueIterationAgent
from planning import GridPlanningModel


PROJECT_ROOT = (Path(__file__).resolve().parent.parent)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def next_model_paths(model_dir: Path,) -> tuple[Path, Path]:
    serial = 1

    while True:
        forage_path = (
            model_dir
            / f"value_iteration_forage_{serial:03d}.pkl"
        )

        chase_path = (
            model_dir
            / f"value_iteration_chase_{serial:03d}.pkl"
        )

        if (
            not forage_path.exists()
            and not chase_path.exists()
        ):
            return forage_path, chase_path

        serial += 1


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

    print("Training foraging policy...")

    forage_agent.train(
        states=model.generate_forage_states(),
        transition_function=(
            model.forage_transitions
        ),
    )

    print("Training chase policy...")

    chase_agent.train(
        states=model.generate_chase_states(),
        transition_function=(
            model.chase_transitions
        ),
    )

    forage_path, chase_path = (
        next_model_paths(MODEL_DIR)
    )

    forage_agent.save(forage_path)
    chase_agent.save(chase_path)

    print(
        f"Saved foraging policy to {forage_path}"
    )

    print(
        f"Saved chase policy to {chase_path}"
    )


if __name__ == "__main__":
    main()