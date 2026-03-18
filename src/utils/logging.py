"""
Experiment logging utilities.

Provides a lightweight wrapper that writes metrics to:
- TensorBoard (via SummaryWriter)
- CSV file (for easy post-hoc analysis)
- Console (structured log lines)

Every experiment run gets its own timestamped directory so prior results
are never overwritten.
"""

from __future__ import annotations

from pathlib import Path

# TODO: implement experiment logger


def make_run_dir(base_dir: str, run_name: str) -> Path:
    """Create a unique timestamped run directory.

    Args:
        base_dir: Parent directory for all runs.
        run_name: Human-readable prefix for the directory name.

    Returns:
        Path to the newly created run directory.
    """
    raise NotImplementedError


class ExperimentLogger:
    """Logs scalar metrics to TensorBoard, CSV, and console."""

    def __init__(self, run_dir: str | Path) -> None:
        raise NotImplementedError

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at the given step."""
        raise NotImplementedError

    def close(self) -> None:
        """Flush and close all writers."""
        raise NotImplementedError
