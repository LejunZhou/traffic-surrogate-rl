"""
Gymnasium environment backed by the trained DeepONet surrogate.

At each step, the agent provides a ramp metering action; the surrogate
predicts the resulting density field; the reward is computed and returned.

Observation (shape (N_x + 2,) = (22,)):
    density[0:N_x]  — z-score normalized density at 20 detector locations
    demand[N_x]     — min-max normalized current mainline demand ∈ [0, 1]
    time[N_x+1]     — normalized time index k / T_ctrl ∈ [0, 1]

Action (shape (1,)):
    ramp metering rate ∈ [0, 1] (continuous Box)

Surrogate rollout strategy (zero-padded partial control sequences):
    At step k, branch input = [u(0),...,u(k), 0,...,0 ; d(0),...,d(T-1)]
    Trunk queries density at all (x_i, t_k).
    DeepONet is re-evaluated from scratch each step (not autoregressive).

See CLAUDE.md "Surrogate rollout strategy" for full rationale and risk notes.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# TODO: implement SurrogateEnv


class SurrogateEnv(gym.Env):
    """Gymnasium environment wrapping a trained DeepONet surrogate."""

    metadata: dict = {"render_modes": []}

    def __init__(self, surrogate_checkpoint: str, config: dict) -> None:
        """
        Args:
            surrogate_checkpoint: Path to trained DeepONet checkpoint (.pt).
            config: Environment config (N_x, T_ctrl, demand_profiles, normalization stats, etc.).
        """
        super().__init__()
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Sample a demand profile and reset the episode state.

        Returns:
            observation: shape (N_x + 2,) = (22,)
            info: dict
        """
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, query surrogate, compute reward, advance time.

        Args:
            action: shape (1,), ramp metering rate ∈ [0, 1]

        Returns:
            observation: shape (N_x + 2,)
            reward: float
            terminated: bool (True when episode ends at T_ctrl steps)
            truncated: bool (always False in Phase 1)
            info: dict
        """
        raise NotImplementedError
