"""
Generate the training dataset by sweeping over demand levels and control signals.

Milestone 2 MVP: constant demand levels only (1000, 1500, 2000 veh/hr).
Time-varying demand profiles and truncated/zero-padded variants are deferred to 2b.

For each simulation:
1. Pick a demand level from the configured set
2. Sample a ramp control signal from the 4-type family
3. Run SUMO via run_simulation()
4. Save the result as data/raw/sim_{index:04d}.npz

Dataset schema (per sample .npz):
    density:             (N_x, T_ctrl)     veh/km — supervised target
    speed:               (N_x, T_ctrl)     km/h   — diagnostic only
    flow:                (N_x, T_ctrl)     veh/hr — diagnostic only
    x_grid:              (N_x,)            detector positions in metres
    t_grid:              (T_ctrl,)         control step timestamps in seconds
    mainline_demand:     (T_ctrl,)         veh/hr (constant in MVP)
    ramp_control:        (120,)            metering rate ∈ [0, 1]
    seed:                ()                int
    mainline_demand_vph: ()                float
    ramp_demand_vph:     ()                float
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sumo_env.network_builder import build_network
from sumo_env.detectors import build_detector_file
from sumo_env.run_simulation import run_simulation
from utils.config import load_config, merge_configs
from utils.plotting import plot_trajectory


# ── Demand profiles ──────────────────────────────────────────────────────────


def make_demand_profile(demand_vph: float, T_ctrl: int) -> np.ndarray:
    """Return a constant demand array for the MVP.

    Args:
        demand_vph: Constant mainline demand in veh/hr.
        T_ctrl: Number of control steps (120).

    Returns:
        shape (T_ctrl,) float32 array filled with demand_vph.
    """
    return np.full(T_ctrl, demand_vph, dtype=np.float32)


# ── Ramp control sampling ───────────────────────────────────────────────────


def sample_ramp_control(
    control_type: str, T_ctrl: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a ramp metering signal from the specified family.

    Args:
        control_type: One of "constant", "piecewise_constant", "smooth",
                      "ramp_step".
        T_ctrl: Number of control steps (120).
        rng: numpy random generator for reproducibility.

    Returns:
        shape (T_ctrl,) float32 array with values in [0, 1].
    """
    if control_type == "constant":
        return _sample_constant(T_ctrl, rng)
    elif control_type == "piecewise_constant":
        return _sample_piecewise_constant(T_ctrl, rng)
    elif control_type == "smooth":
        return _sample_smooth(T_ctrl, rng)
    elif control_type == "ramp_step":
        return _sample_ramp_step(T_ctrl, rng)
    else:
        raise ValueError(f"Unknown ramp control type: {control_type!r}")


def _sample_constant(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(0.0, 1.0)
    return np.full(T_ctrl, r, dtype=np.float32)


def _sample_piecewise_constant(
    T_ctrl: int, rng: np.random.Generator
) -> np.ndarray:
    n_segments = rng.integers(2, 7)  # 2 to 6 segments
    changepoints = np.sort(rng.integers(1, T_ctrl, size=n_segments - 1))
    levels = rng.uniform(0.0, 1.0, size=n_segments).astype(np.float32)

    signal = np.empty(T_ctrl, dtype=np.float32)
    prev = 0
    for i, cp in enumerate(changepoints):
        signal[prev:cp] = levels[i]
        prev = cp
    signal[prev:] = levels[-1]
    return signal


def _sample_smooth(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    n_knots = rng.integers(4, 9)  # 4 to 8 knots
    knot_positions = np.linspace(0, T_ctrl - 1, n_knots)
    knot_values = rng.uniform(0.0, 1.0, size=n_knots)
    signal = np.interp(
        np.arange(T_ctrl), knot_positions, knot_values
    ).astype(np.float32)
    return np.clip(signal, 0.0, 1.0)


def _sample_ramp_step(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    peak = rng.uniform(0.3, 1.0)
    pattern = rng.integers(0, 4)  # 4 patterns

    # Randomize transition timing
    t_start = rng.integers(T_ctrl // 6, T_ctrl // 3)
    t_end = rng.integers(2 * T_ctrl // 3, 5 * T_ctrl // 6)

    signal = np.zeros(T_ctrl, dtype=np.float32)

    if pattern == 0:  # monotone increase: 0 → peak
        signal[:t_start] = 0.0
        ramp = np.linspace(0.0, peak, t_end - t_start, dtype=np.float32)
        signal[t_start:t_end] = ramp
        signal[t_end:] = peak
    elif pattern == 1:  # monotone decrease: peak → 0
        signal[:t_start] = peak
        ramp = np.linspace(peak, 0.0, t_end - t_start, dtype=np.float32)
        signal[t_start:t_end] = ramp
        signal[t_end:] = 0.0
    elif pattern == 2:  # up-then-down: 0 → peak → 0
        mid = (t_start + t_end) // 2
        signal[:t_start] = 0.0
        signal[t_start:mid] = np.linspace(
            0.0, peak, mid - t_start, dtype=np.float32
        )
        signal[mid:t_end] = np.linspace(
            peak, 0.0, t_end - mid, dtype=np.float32
        )
        signal[t_end:] = 0.0
    else:  # down-then-up: peak → 0 → peak
        mid = (t_start + t_end) // 2
        signal[:t_start] = peak
        signal[t_start:mid] = np.linspace(
            peak, 0.0, mid - t_start, dtype=np.float32
        )
        signal[mid:t_end] = np.linspace(
            0.0, peak, t_end - mid, dtype=np.float32
        )
        signal[t_end:] = peak

    return np.clip(signal, 0.0, 1.0)


# ── Dataset generation ───────────────────────────────────────────────────────


def generate_dataset(
    dataset_config_path: str, project_root: Path | None = None
) -> list[Path]:
    """Run all simulations and save raw trajectory files.

    Args:
        dataset_config_path: Path to the dataset experiment config YAML.
        project_root: Project root directory. If None, inferred from
                      this file's location (../../).

    Returns:
        List of paths to saved .npz files.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    ds_config = load_config(dataset_config_path)
    base_sumo_config = load_config(
        str(project_root / ds_config["base_sumo_config"])
    )

    ds = ds_config["dataset"]
    out = ds_config["output"]
    n_samples: int = ds["n_samples"]
    seed: int = ds["random_seed"]
    demand_levels: list[float] = ds["demand_levels"]
    control_types: list[str] = ds["ramp_control_types"]
    save_heatmaps: bool = out.get("save_heatmaps", False)
    heatmap_every_n: int = out.get("heatmap_every_n", 10)

    raw_dir = project_root / out["raw_dir"]
    network_dir = project_root / out["network_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    sim_cfg = base_sumo_config["simulation"]
    T_ctrl = int(sim_cfg["duration_s"] / sim_cfg["dt_ctrl_s"])

    # Build network topology + detectors once (reused across all runs).
    # Routes will be rebuilt per demand level.
    print(f"[dataset] Building network in {network_dir} ...")
    network_files = build_network(str(network_dir), base_sumo_config)
    det_file = str(network_dir / "detectors.add.xml")
    build_detector_file(det_file, base_sumo_config)

    # Pre-build routes for each demand level to avoid repeated rebuilds
    routes_by_demand: dict[float, str] = {}
    for demand_vph in demand_levels:
        cfg_for_demand = merge_configs(
            base_sumo_config, {"demand": {"mainline_demand_vph": demand_vph}}
        )
        route_path = network_dir / f"routes_{int(demand_vph)}.rou.xml"
        from sumo_env.network_builder import _write_routes

        _write_routes(route_path, cfg_for_demand)
        routes_by_demand[demand_vph] = str(route_path.resolve())
        print(f"[dataset] Routes for {demand_vph} vph → {route_path.name}")

    saved_paths: list[Path] = []
    total_teleports = 0

    print(f"[dataset] Generating {n_samples} samples ...")
    for i in range(n_samples):
        # Round-robin demand levels, random control type
        demand_vph = demand_levels[i % len(demand_levels)]
        control_type = control_types[i % len(control_types)]

        ramp_control = sample_ramp_control(control_type, T_ctrl, rng)

        # Build config with the chosen demand level
        sim_config = merge_configs(
            base_sumo_config,
            {
                "demand": {"mainline_demand_vph": demand_vph},
                "simulation": {"seed": seed + i},
            },
        )

        result = run_simulation(
            net_file=network_files["net"],
            route_file=routes_by_demand[demand_vph],
            detector_file=det_file,
            ramp_control=ramp_control,
            config=sim_config,
        )

        teleports = result["metadata"]["teleports"]
        total_teleports += teleports

        # Save .npz
        traj_path = raw_dir / f"sim_{i:04d}.npz"
        np.savez(
            str(traj_path),
            density=result["density"],
            speed=result["speed"],
            flow=result["flow"],
            x_grid=result["x_grid"],
            t_grid=result["t_grid"],
            mainline_demand=result["mainline_demand"],
            ramp_control=result["ramp_control"],
            seed=np.array(sim_config["simulation"]["seed"]),
            mainline_demand_vph=np.array(demand_vph),
            ramp_demand_vph=np.array(
                result["metadata"]["ramp_demand_vph"]
            ),
        )
        saved_paths.append(traj_path)

        # Optional heatmap
        if save_heatmaps and i % heatmap_every_n == 0:
            plot_path = raw_dir / f"sim_{i:04d}_density.png"
            plot_trajectory(
                density=result["density"],
                x_grid=result["x_grid"],
                t_grid=result["t_grid"],
                output_path=plot_path,
                title=(
                    f"sim_{i:04d} — {int(demand_vph)} vph, "
                    f"{control_type}, seed={sim_config['simulation']['seed']}"
                ),
            )

        status = "OK" if teleports == 0 else f"TELEPORTS={teleports}"
        inserts = result["metadata"]["insert_success"]
        attempts = result["metadata"]["insert_attempts"]
        print(
            f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] "
            f"demand={int(demand_vph):>4}, ctrl={control_type:<20s}, "
            f"inserts={inserts}/{attempts}, {status}"
        )

    print(f"\n[dataset] Done. {n_samples} samples saved to {raw_dir}")
    if total_teleports > 0:
        print(f"[dataset] WARNING: {total_teleports} total teleports detected!")
    else:
        print("[dataset] All simulations had 0 teleports.")

    return saved_paths


# ── Train/val/test splits ────────────────────────────────────────────────────


def make_splits(
    raw_dir: str,
    splits_dir: str,
    config: dict,
    seed: int = 42,
) -> dict:
    """Aggregate raw .npz files and write train/val/test splits.

    Splits are over base trajectories. Normalization statistics are
    computed from the training set only.

    Args:
        raw_dir: Directory containing sim_*.npz files.
        splits_dir: Output directory for split index files and metadata.
        config: Split config with keys train_frac, val_frac, test_frac.
        seed: Random seed for shuffling.

    Returns:
        Dict with keys "train", "val", "test" (lists of filenames)
        and "metadata" (normalization stats).
    """
    raw_path = Path(raw_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    # Find all base samples (not truncated)
    npz_files = sorted(raw_path.glob("sim_[0-9][0-9][0-9][0-9].npz"))
    if not npz_files:
        raise FileNotFoundError(f"No sim_*.npz files found in {raw_dir}")

    n = len(npz_files)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    train_frac = config["train_frac"]
    val_frac = config["val_frac"]
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac)) - n_train

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {
        "train": [npz_files[i].name for i in train_idx],
        "val": [npz_files[i].name for i in val_idx],
        "test": [npz_files[i].name for i in test_idx],
    }

    # Compute normalization stats from training set only
    train_densities = []
    train_demands = []
    for fname in splits["train"]:
        data = np.load(str(raw_path / fname))
        train_densities.append(data["density"])
        train_demands.append(float(data["mainline_demand_vph"]))

    all_density = np.concatenate(
        [d.ravel() for d in train_densities]
    )
    mean_density = float(np.mean(all_density))
    std_density = float(np.std(all_density))

    # Min-max for demand across the training set
    min_demand = float(min(train_demands))
    max_demand = float(max(train_demands))

    metadata = {
        "mean_density": mean_density,
        "std_density": std_density,
        "min_demand": min_demand,
        "max_demand": max_demand,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_total": n,
        "seed": seed,
    }

    # Save split indices
    split_index = {**splits, "metadata": metadata}
    index_path = splits_path / "split_index.json"
    with open(index_path, "w") as f:
        json.dump(split_index, f, indent=2)

    # Save metadata separately for easy loading
    meta_path = splits_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[splits] train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"[splits] density: mean={mean_density:.3f}, std={std_density:.3f}")
    print(f"[splits] demand:  min={min_demand:.0f}, max={max_demand:.0f}")
    print(f"[splits] Saved to {splits_path}")

    return split_index


# ── CLI entry point ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT / "src"))

    parser = argparse.ArgumentParser(
        description="Generate SUMO dataset for DeepONet training"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to dataset experiment config YAML",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override n_samples from config (for smoke tests)",
    )
    args = parser.parse_args()

    config_path = str(_PROJECT_ROOT / args.config)
    ds_cfg = load_config(config_path)

    # Allow CLI override of n_samples for quick smoke tests
    if args.n_samples is not None:
        ds_cfg["dataset"]["n_samples"] = args.n_samples

    # Write back the potentially modified config for generate_dataset
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(ds_cfg, tmp)
        tmp_path = tmp.name

    saved = generate_dataset(tmp_path, project_root=_PROJECT_ROOT)

    # Run splits
    out = ds_cfg["output"]
    splits = ds_cfg["splits"]
    make_splits(
        raw_dir=str(_PROJECT_ROOT / out["raw_dir"]),
        splits_dir=str(_PROJECT_ROOT / out["splits_dir"]),
        config=splits,
        seed=ds_cfg["dataset"]["random_seed"],
    )

    Path(tmp_path).unlink(missing_ok=True)
