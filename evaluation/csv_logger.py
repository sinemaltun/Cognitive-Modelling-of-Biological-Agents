import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, output_dir: Path,):
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True,exist_ok=True,)

    def log_step(self, record) -> None:
        self._append("steps.csv",record.to_dict(),)

    def log_episode(self, record) -> None:
        self._append("episodes.csv",record.to_dict())

    def log_training_progress(self, record,) -> None:
        self._append("training_progress.csv", record.to_dict(),)

    def log_value_iteration_progress(self, record,) -> None:
        self._append("value_iteration_progress.csv", record.to_dict(),)

    def _append(self, filename: str, row: dict,) -> None:
        path = self.output_dir / filename

        write_header = (
            not path.exists()
            or path.stat().st_size == 0
        )

        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()),)

            if write_header:
                writer.writeheader()

            writer.writerow(row)