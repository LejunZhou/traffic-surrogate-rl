"""
Experiment logging utilities.

Provides a lightweight wrapper that writes metrics to:
- Weights & Biases, when enabled and installed
- CSV file (for easy post-hoc analysis)
- Console (structured log lines)

Every experiment run gets its own timestamped directory so prior results
are never overwritten.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at runtime
    wandb = None


def make_run_dir(base_dir: str, run_name: str) -> Path:
    """Create a unique timestamped run directory.

    Args:
        base_dir: Parent directory for all runs.
        run_name: Human-readable prefix for the directory name.

    Returns:
        Path to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class ExperimentLogger:
    """Logs scalar metrics to W&B, CSV, and console."""

    def __init__(
        self,
        run_dir: str | Path,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer: csv.DictWriter | None = None
        self.fieldnames: list[str] | None = None
        self.wandb_run = None

        if use_wandb:
            if wandb is None:
                raise ImportError(
                    "wandb logging was enabled, but the 'wandb' package is not installed. "
                    "Install it with `pip install wandb` or set output.wandb.enabled=false."
                )
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
                dir=str(self.run_dir),
            )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at the given step."""
        row = {"step": step, **metrics}
        if self.csv_writer is None:
            self.fieldnames = list(row.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

        pretty = ", ".join(f"{k}={v:.6g}" for k, v in metrics.items())
        print(f"[metrics] step={step} {pretty}")

    def close(self) -> None:
        """Flush and close all writers."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self.csv_file.close()
