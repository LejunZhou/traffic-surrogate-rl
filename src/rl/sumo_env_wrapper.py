"""
Gymnasium environment backed by live SUMO simulation via TraCI.

Exposes the same observation_space, action_space, and reward function as
SurrogateEnv so that PPO training code is fully environment-agnostic.

Observation (shape (N_x + 2,) = (22,)):
    density[0:N_x]  — z-score normalized density at 20 detector locations
    demand[N_x]     — min-max normalized current mainline demand ∈ [0, 1]
    time[N_x+1]     — normalized time index k / T_ctrl ∈ [0, 1]

Action (shape (1,)):
    ramp metering rate ∈ [0, 1] (continuous Box)

At each step:
1. Apply ramp metering rate via TraCI (override on-ramp flow)
2. Advance SUMO by dt_ctrl simulation seconds
3. Read E2 detector aggregates → density field
4. Compute reward via reward.compute_reward()
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# TODO: implement SumoEnv


class SumoEnv(gym.Env):
    """Gymnasium environment wrapping SUMO via TraCI."""

    metadata: dict = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Environment config (net_file, route_file, detector_file,
                    dt_ctrl, T_ctrl, N_x, demand_profiles, normalization stats,
                    sumo_binary, seed, etc.).
        """
        super().__init__()
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new SUMO simulation with a sampled demand profile.

        Returns:
            observation: shape (N_x + 2,) = (22,)
            info: dict
        """
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply ramp metering, advance SUMO, read detectors, compute reward.

        Args:
            action: shape (1,), ramp metering rate ∈ [0, 1]

        Returns:
            observation: shape (N_x + 2,)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the TraCI connection and terminate SUMO."""
        raise NotImplementedError
