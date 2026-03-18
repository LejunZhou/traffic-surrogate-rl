"""
Plotting utilities for surrogate evaluation and RL training analysis.

Functions:
- plot_trajectory:    single-trajectory density heatmap (x vs t) — Milestone 1
- plot_density_heatmap: predicted-vs-true density heatmap (x vs t) — Milestone 3+
- plot_reward_curve: PPO training reward over environment steps
- plot_comparison_bar: comparison bar chart across policies and metrics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_trajectory(
    density: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    output_path: str | Path,
    title: str = "Traffic density — SUMO rollout",
) -> None:
    """Plot a single density trajectory as a space–time heatmap.

    Used in Milestone 1 as the diagnostic output for one simulation rollout.
    x-axis = time [s], y-axis = position [m], colour = density [veh/km].

    Args:
        density:     shape (N_x, T_ctrl) — density in veh/km
        x_grid:      shape (N_x,)  — detector positions in metres
        t_grid:      shape (T_ctrl,) — time points in seconds
        output_path: Save location for the plot (.png).
        title:       Plot title string.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    t_min, t_max = float(t_grid[0]), float(t_grid[-1])
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])

    im = ax.imshow(
        density,
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, x_min, x_max],
        cmap="hot_r",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density [veh/km]", fontsize=11)

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position [m]", fontsize=11)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# TODO: implement plotting functions (Milestones 3–7)


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
