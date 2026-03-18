"""
Evaluate a trained PPO policy in SUMO.

Always evaluates in SUMO (the ground-truth simulator), regardless of which
environment the policy was trained in.

Outputs:
- Per-episode metrics: total reward, mean density, throughput, queue length
- Comparison table: surrogate-trained policy vs. SUMO-trained policy
- Trajectory plots for qualitative analysis
"""

from __future__ import annotations

import numpy as np

# TODO: implement policy evaluation


def evaluate_in_sumo(policy_path: str, config: dict) -> dict:
    """Roll out a trained PPO policy in SUMO for n_episodes and collect metrics.

    Args:
        policy_path: Path to a saved SB3 model (.zip).
        config: Evaluation config (sumo env config, n_episodes, output_dir, etc.).

    Returns:
        Dict of metrics:
            "mean_total_reward": float
            "mean_density":      float   (lower = less congestion)
            "throughput":        float   (veh/hr exiting the segment)
            "mean_queue_length": float   (on-ramp queue)
            "episodes":          list[dict]  (per-episode breakdown)
    """
    raise NotImplementedError
