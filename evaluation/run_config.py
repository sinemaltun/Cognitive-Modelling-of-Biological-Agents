import json
from pathlib import Path
from typing import Any


def _save_json(path: Path, data: dict[str, Any],) -> None:
    path.parent.mkdir(parents=True, exist_ok=True,)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def save_run_config(output_dir: Path, config: dict[str, Any],) -> None:
    """
    Save the settings that existed before a run started.
    """
    output_dir = Path(output_dir)

    _save_json(output_dir / "run_config.json", config,)


def save_run_summary(output_dir: Path, summary: dict[str, Any],) -> None:
    """
    Save final information after a run has completed.
    """
    output_dir = Path(output_dir)

    _save_json(output_dir / "run_summary.json", summary,)