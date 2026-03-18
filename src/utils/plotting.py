"""
Plotting utilities for surrogate evaluation and RL training analysis.

Functions:
- plot_density_heatmap: predicted-vs-true density heatmap (x vs t)
- plot_reward_curve: PPO training reward over environment steps
- plot_comparison_bar: comparison bar chart across policies and metrics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TODO: implement plotting functions


def plot_density_heatmap(
    predicted: np.ndarray,
    true: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot side-by-side predicted and true density heatmaps.

    Args:
        predicted: shape (N_x, T_ctrl) — surrogate predictions (physical units)
        true:      shape (N_x, T_ctrl) — SUMO ground truth (physical units)
        x_grid:    shape (N_x,) — detector positions in metres
        t_grid:    shape (T_ctrl,) — time points in seconds
        output_path: Save location for the plot (.png).
    """
    raise NotImplementedError


def plot_reward_curve(
    rewards: list[float],
    output_path: str | Path,
    label: str = "PPO",
) -> None:
    """Plot episode reward over training steps.

    Args:
        rewards: List of per-episode total rewards.
        output_path: Save location for the plot (.png).
        label: Legend label for the curve.
    """
    raise NotImplementedError


def plot_comparison_bar(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Plot grouped bar chart comparing metrics across policies.

    Args:
        results: {policy_name: {metric_name: value, ...}, ...}
        output_path: Save location for the plot (.png).
    """
    raise NotImplementedError
