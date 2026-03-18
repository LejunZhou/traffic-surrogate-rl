"""
Evaluation of a trained DeepONet surrogate.

Produces:
- Per-sample L2 error and relative L2 error on the test split
- Predicted-vs-true density heatmaps (x vs t) for a selection of test samples
- Summary metrics saved to eval_metrics.json in the run directory
"""

from __future__ import annotations

import numpy as np

# TODO: implement surrogate evaluation


def evaluate(checkpoint_path: str, config: dict) -> dict:
    """Evaluate a trained DeepONet on the test split.

    Args:
        checkpoint_path: Path to a saved model checkpoint (.pt file).
        config: Evaluation config (data paths, plot settings, output dir).

    Returns:
        Dict of scalar metrics: {"mean_l2": ..., "rel_l2": ..., ...}
    """
    raise NotImplementedError
