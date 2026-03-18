"""
Generate the training dataset by sweeping over demand profiles and control signals.

For each simulation:
1. Sample a demand profile from the Phase 1 controlled family
2. Sample a random ramp metering signal
3. Run SUMO via run_simulation()
4. Save the result as data/raw/sim_{index:04d}.npz

Also generates truncated/zero-padded variants per CLAUDE.md dataset design requirement:
- For each full trajectory, create additional samples truncated at random cut points
- Query points for truncated samples are restricted to t <= t_k
- This is required for the surrogate to generalize to RL rollout conditions.

Dataset schema (per sample .npz):
    mainline_demand: (T_ctrl,)         veh/hr, one of the controlled demand profiles
    ramp_control:    (T_ctrl,)         metering rate ∈ [0, 1]
    density:         (N_x, T_ctrl)     veh/km — supervised target
    speed:           (N_x, T_ctrl)     km/h   — logged for diagnostics only
    flow:            (N_x, T_ctrl)     veh/hr — logged for diagnostics only
    x_grid:          (N_x,)            detector positions in metres
    t_grid:          (T_ctrl,)         control step timestamps in seconds
    seed:            int
    demand_profile:  str               label of the demand profile used
"""

from __future__ import annotations

# TODO: implement dataset generation sweep


def generate_dataset(config: dict) -> None:
    """Run all simulations and save raw trajectory files.

    Args:
        config: Full experiment config. Expected keys include:
            n_samples (int), output_dir (str), demand_profiles (list),
            ramp_control_sampling (str), seed (int), sumo config keys.
    """
    raise NotImplementedError


def make_splits(raw_dir: str, splits_dir: str, config: dict) -> None:
    """Aggregate raw .npz files and write train/val/test splits.

    Exact split logic, normalization strategy, and output format
    to be finalized in Milestone 2.

    Args:
        raw_dir: Directory containing sim_*.npz files.
        splits_dir: Output directory for split files and metadata.
        config: Split config (to be specified in Milestone 2).
    """
    raise NotImplementedError
