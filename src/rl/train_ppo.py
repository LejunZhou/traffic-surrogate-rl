"""
Train a PPO policy using Stable-Baselines3.

The environment is selected from config: either SurrogateEnv or SumoEnv.
Both expose identical observation/action/reward interfaces, so this script
is fully agnostic to which backend is used.

Outputs (saved to a timestamped run directory):
- PPO model checkpoint (best + final)
- Training reward curve (TensorBoard + CSV)
- Config snapshot and random seed
"""

from __future__ import annotations

# TODO: implement PPO training entry point


def train(config: dict) -> None:
    """Train PPO on the configured environment.

    Args:
        config: Training config. Expected keys include:
            env.type ("surrogate" | "sumo"),
            env (environment-specific sub-config),
            ppo (SB3 PPO hyperparameters: n_steps, batch_size, n_epochs, lr, ...),
            training (total_timesteps, eval_freq, seed),
            output.run_dir
    """
    raise NotImplementedError
