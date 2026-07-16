import json
from pathlib import Path
from typing import Any


def save_run_config(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "run_config.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            default=str,
        )