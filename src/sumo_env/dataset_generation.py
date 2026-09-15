"""
Generate base SUMO rollouts by sweeping over demand levels and control signals.

Milestone 2 MVP: constant demand levels only (1000, 1500, 2000 veh/hr).
Time-varying demand profiles are deferred to 2b. Truncated/zero-padded
control views are generated later by surrogate.datasets.TrafficDataset so the
same physical rollout can supervise both full-control and RL-style partial
control inputs without rerunning SUMO.

For each simulation:
1. Pick a demand level from the configured set
2. Sample a ramp control signal from the 4-type family
3. Run SUMO via run_simulation()
4. Save the result under output.raw_dir as sim_{index:04d}.npz

Dataset schema (per sample .npz):
    density:             (N_x, T_ctrl)     veh/km — supervised target
    speed:               (N_x, T_ctrl)     km/h   — diagnostic only
    flow:                (N_x, T_ctrl)     veh/hr — diagnostic only
    exit_boundary_flow_vph: (T_ctrl,)      veh/hr — mean of last 3 detector flows at each timestep
    x_grid:              (N_x,)            detector positions in metres
    t_grid:              (T_ctrl,)         control step timestamps in seconds
    mainline_demand:     (T_ctrl,)         veh/hr (constant in MVP)
    ramp_control:        (T_ctrl,)         confirmed inflow / reference (command in open_loop)
    ramp_control_cmd:    (T_ctrl,)         requested metering rate ∈ [0, 1]
    ramp_inflow_vph:     (T_ctrl,)         confirmed ramp entries / interval hours
    ramp_queue:          (T_ctrl,)         arrivals minus confirmed entries (0 in open_loop)
    ramp_departed_count: (T_ctrl,)         confirmed ramp entries per interval
    ramp_pending_count:  (T_ctrl,)         pending ramp requests at interval end (included in metered queue)
    ramp_flow_measurement: ()              "confirmed_departures"
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
    start_index: int = int(ds.get("start_index", 0))
    overwrite: bool = bool(ds.get("overwrite", False))
    demand_levels: list[float] = ds["demand_levels"]
    # Ramp arrival rates (vph) cycled per sample alongside the mainline levels;
    # default = the scenario's demand.ramp_demand_vph (single level).
    ramp_demand_levels: list[float] = [
        float(v) for v in ds.get("ramp_demand_levels", [base_sumo_config["demand"]["ramp_demand_vph"]])
    ]
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

    if start_index > 0:
        _advance_control_rng(rng, control_types, T_ctrl, start_index)

    print(
        f"[dataset] Generating {n_samples} samples "
        f"(indices {start_index:04d}..{start_index + n_samples - 1:04d}) ..."
    )
    for i in range(n_samples):
        sample_index = start_index + i
        # Round-robin over mainline levels, then ramp levels (so every
        # (mainline, ramp) cell is visited equally often), random control type
        demand_vph = demand_levels[sample_index % len(demand_levels)]
        ramp_demand_vph = ramp_demand_levels[(sample_index // len(demand_levels)) % len(ramp_demand_levels)]
        control_type = control_types[sample_index % len(control_types)]

        ramp_control = sample_ramp_control(control_type, T_ctrl, rng)

        # Build config with the chosen demand level
        sim_config = merge_configs(
            base_sumo_config,
            {
                "demand": {"mainline_demand_vph": demand_vph, "ramp_demand_vph": ramp_demand_vph},
                "simulation": {"seed": seed + sample_index},
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
        traj_path = raw_dir / f"sim_{sample_index:04d}.npz"
        if traj_path.exists() and not overwrite:
            raise FileExistsError(
                f"{traj_path} already exists. Use --append to continue after "
                "existing files, --start-index to choose a different index, "
                "or --overwrite to replace existing files."
            )
        np.savez(
            str(traj_path),
            density=result["density"],
            speed=result["speed"],
            flow=result["flow"],
            exit_boundary_flow_vph=result["exit_boundary_flow_vph"],
            x_grid=result["x_grid"],
            t_grid=result["t_grid"],
            mainline_demand=result["mainline_demand"],
            ramp_control=result["ramp_control"],
            ramp_control_cmd=result["ramp_control_cmd"],
            ramp_inflow_vph=result["ramp_inflow_vph"],
            ramp_queue=result["ramp_queue"],
            ramp_departed_count=result["ramp_departed_count"],
            ramp_pending_count=result["ramp_pending_count"],
            ramp_flow_measurement=np.array(result["metadata"]["ramp_flow_measurement"]),
            ramp_model=np.array(result["metadata"]["ramp_model"]),
            ramp_ref_vph=np.array(result["metadata"]["ramp_ref_vph"]),
            ramp_discharge_vph=np.array(result["metadata"]["ramp_discharge_vph"]),
            seed=np.array(sim_config["simulation"]["seed"]),
            mainline_demand_vph=np.array(demand_vph),
            ramp_demand_vph=np.array(
                result["metadata"]["ramp_demand_vph"]
            ),
        )
        saved_paths.append(traj_path)

        # Optional heatmap
        if save_heatmaps and i % heatmap_every_n == 0:
            plot_path = raw_dir / f"sim_{sample_index:04d}_density.png"
            plot_trajectory(
                density=result["density"],
                x_grid=result["x_grid"],
                t_grid=result["t_grid"],
                output_path=plot_path,
                title=(
                    f"sim_{sample_index:04d} — {int(demand_vph)} vph, "
                    f"{control_type}, seed={sim_config['simulation']['seed']}"
                ),
            )

        status = "OK" if teleports == 0 else f"TELEPORTS={teleports}"
        inserts = result["metadata"]["insert_success"]
        attempts = result["metadata"]["insert_attempts"]
        departed = result["metadata"]["ramp_departed_total"]
        print(
            f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] "
            f"demand={int(demand_vph):>4}+{int(ramp_demand_vph):<3}, ctrl={control_type:<20s}, "
            f"inflow_max={result['metadata']['ramp_inflow_max_vph']:4.0f}vph, "
            f"requests={inserts}/{attempts}, entered={departed}, {status}"
        )

    print(f"\n[dataset] Done. {n_samples} samples saved to {raw_dir}")
    if total_teleports > 0:
        print(f"[dataset] WARNING: {total_teleports} total teleports detected!")
    else:
        print("[dataset] All simulations had 0 teleports.")

    return saved_paths


def next_sample_index(raw_dir: str | Path) -> int:
    """Return one plus the largest sim_*.npz index in raw_dir."""
    raw_path = Path(raw_dir)
    max_index = -1
    for path in raw_path.glob("sim_*.npz"):
        suffix = path.stem.removeprefix("sim_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _advance_control_rng(
    rng: np.random.Generator,
    control_types: list[str],
    T_ctrl: int,
    n_steps: int,
) -> None:
    """Advance control RNG so appended runs match one contiguous generation."""
    for i in range(n_steps):
        control_type = control_types[i % len(control_types)]
        sample_ramp_control(control_type, T_ctrl, rng)


# ── Family-schema generation plan (M8 step 1: brought into the tree) ─────────
#
# The shockwave dataset (M3, other machine) used a "family" schema in the
# experiment config (dataset.families: [{name, count, params}]) whose per-rollout
# RNG is addressed by (base_seed, family_id, local_index), so any partition of
# the plan reproduces the same rollouts (scripts/run_parallel_generation.py).
# Open-loop control families: constant_grid, bang_bang, piecewise_constant,
# fourier, ramp_step, smooth. Files are named <family>_<local_index:05d>.npz.

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationSpec:
    family_id: int
    name: str
    local_index: int
    count: int
    params: dict


def build_generation_plan(families: list[dict]) -> list[GenerationSpec]:
    plan: list[GenerationSpec] = []
    for fid, fam in enumerate(families):
        for j in range(int(fam["count"])):
            plan.append(GenerationSpec(fid, str(fam["name"]), j, int(fam["count"]), dict(fam.get("params", {}) or {})))
    return plan


def _dist(rng: np.random.Generator, spec, size=None):
    """Level distribution: [lo, hi] uniform, or 'beta_a' (symmetric Beta(a, a) on [0, 1])."""
    if isinstance(spec, str) and spec.startswith("beta_"):
        a = float(spec.split("_", 1)[1])
        return rng.beta(a, a, size=size)
    lo, hi = float(spec[0]), float(spec[1])
    return rng.uniform(lo, hi, size=size)


def sample_family_control(spec: GenerationSpec, T_ctrl: int, base_seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), spec.family_id, spec.local_index]))
    prm = spec.params
    name = spec.name
    if name == "constant_grid":
        return np.full(T_ctrl, float(np.linspace(0.0, 1.0, spec.count)[spec.local_index]), dtype=np.float32)
    if name == "bang_bang":
        lo, hi = prm.get("duty", [0.2, 0.8])
        n_sw = int(rng.integers(1, int(prm.get("max_switches", 12)) + 1))
        min_dwell = int(prm.get("min_dwell", 2)); max_period = int(prm.get("max_period", 40))
        u_hi = float(rng.uniform(0.6, 1.0)); u_lo = float(rng.uniform(0.0, 0.4))
        sig = np.empty(T_ctrl, np.float32); k = 0; level = bool(rng.integers(0, 2)); switches = 0
        while k < T_ctrl:
            dwell = int(rng.integers(min_dwell, max(min_dwell + 1, max_period)))
            sig[k:k + dwell] = u_hi if level else u_lo
            k += dwell; switches += 1
            if switches < n_sw:
                level = not level
        return np.clip(sig, 0, 1)
    if name == "piecewise_constant":
        lo, hi = prm.get("n_seg", [2, 13]); n_seg = int(rng.integers(int(lo), int(hi)))
        min_seg = int(prm.get("min_seg", 2))
        cuts = np.sort(rng.choice(np.arange(min_seg, T_ctrl - min_seg), size=n_seg - 1, replace=False)) if n_seg > 1 else np.array([], int)
        levels = _dist(rng, prm.get("level_dist", [0.0, 1.0]), size=n_seg)
        sig = np.empty(T_ctrl, np.float32); prev = 0
        for i, c in enumerate(list(cuts) + [T_ctrl]):
            sig[prev:c] = levels[i]; prev = c
        return np.clip(sig, 0, 1)
    if name == "fourier":
        n_modes = int(prm.get("n_modes", 5)); base = _dist(rng, prm.get("base", [0.3, 0.7]))
        amps = np.asarray(prm.get("amplitudes", [0.3, 0.2, 0.13, 0.08, 0.06]))[:n_modes]
        gain = _dist(rng, prm.get("gain", [1.0, 2.0]))
        t = np.arange(T_ctrl) / T_ctrl
        sig = base + sum(gain * amps[m] * np.sin(2 * np.pi * (m + 1) * t + rng.uniform(0, 2 * np.pi)) for m in range(n_modes))
        return np.clip(sig, 0, 1).astype(np.float32)
    if name == "ramp_step":
        sig = _sample_ramp_step(T_ctrl, rng)
        lo, hi = prm.get("peak", [0.3, 1.0])
        return np.clip(sig * float(rng.uniform(lo, hi)) / max(sig.max(), 1e-6), 0, 1).astype(np.float32)
    if name == "smooth":
        lo, hi = prm.get("n_knots", [3, 11]); n_knots = int(rng.integers(int(lo), int(hi)))
        knots = _dist(rng, prm.get("level_dist", [0.0, 1.0]), size=n_knots)
        return np.clip(np.interp(np.arange(T_ctrl), np.linspace(0, T_ctrl - 1, n_knots), knots), 0, 1).astype(np.float32)
    raise ValueError(f"unknown control family {name!r}")


def _resolve_network_files(network_dir: Path, base_sumo_config: dict, demand_levels: list, reuse_network: bool = False) -> tuple[dict, str, dict]:
    """Build (or reuse) the network, detectors and one route file per demand level."""
    network_dir = Path(network_dir)
    net_path = network_dir / "net.net.xml"
    if reuse_network and net_path.exists():
        network_files = {"net": str(net_path.resolve()), "route": str((network_dir / "routes.rou.xml").resolve())}
    else:
        network_files = build_network(str(network_dir), base_sumo_config)
    det_file = str(network_dir / "detectors.add.xml")
    if not (reuse_network and Path(det_file).exists()):
        build_detector_file(det_file, base_sumo_config)
    from sumo_env.network_builder import _write_routes

    routes = {}
    for demand_vph in demand_levels:
        route_path = network_dir / f"routes_{int(demand_vph)}.rou.xml"
        if not (reuse_network and route_path.exists()):
            _write_routes(route_path, merge_configs(base_sumo_config, {"demand": {"mainline_demand_vph": float(demand_vph)}}))
        routes[float(demand_vph)] = str(route_path.resolve())
    return network_files, det_file, routes


def generate_family_dataset(ds_config: dict, project_root: Path, indices: list[int] | None = None,
                            reuse_network: bool = False, overwrite: bool = False) -> list[Path]:
    """Family-schema generation (open-loop controls, constant demand cells)."""
    base_sumo_config = load_config(str(project_root / ds_config["base_sumo_config"]))
    ds, out = ds_config["dataset"], ds_config["output"]
    raw_dir = project_root / out["raw_dir"]; raw_dir.mkdir(parents=True, exist_ok=True)
    network_dir = project_root / out["network_dir"]
    demand_levels = [float(v) for v in ds.get("demand_levels", [base_sumo_config["demand"]["mainline_demand_vph"]])]
    ramp_levels = [float(v) for v in ds.get("ramp_demand_levels", [base_sumo_config["demand"]["ramp_demand_vph"]])]
    base_seed = int(ds.get("base_seed", ds.get("random_seed", 42)))
    plan = build_generation_plan(ds["families"])
    network_files, det_file, routes = _resolve_network_files(network_dir, base_sumo_config, demand_levels, reuse_network)
    sim_cfg = base_sumo_config["simulation"]
    T_ctrl = int(sim_cfg["duration_s"] / sim_cfg["dt_ctrl_s"])
    todo = range(len(plan)) if indices is None else [int(i) for i in indices]
    saved = []
    for g in todo:
        spec = plan[g]
        path = raw_dir / f"{spec.name}_{spec.local_index:05d}.npz"
        if path.exists() and not overwrite:
            continue
        demand_vph = demand_levels[g % len(demand_levels)]
        ramp_vph = ramp_levels[(g // len(demand_levels)) % len(ramp_levels)]
        control = sample_family_control(spec, T_ctrl, base_seed)
        sim_config = merge_configs(base_sumo_config, {"demand": {"mainline_demand_vph": demand_vph, "ramp_demand_vph": ramp_vph},
                                                      "simulation": {"seed": base_seed + g}})
        result = run_simulation(network_files["net"], routes[demand_vph], det_file, control, sim_config)
        np.savez(str(path), density=result["density"], speed=result["speed"], flow=result["flow"],
                 exit_boundary_flow_vph=result["exit_boundary_flow_vph"], x_grid=result["x_grid"], t_grid=result["t_grid"],
                 mainline_demand=result["mainline_demand"], ramp_control=result["ramp_control"], ramp_control_cmd=result["ramp_control_cmd"],
                 ramp_inflow_vph=result["ramp_inflow_vph"], ramp_queue=result["ramp_queue"], ramp_departed_count=result["ramp_departed_count"],
                 ramp_pending_count=result["ramp_pending_count"], ramp_flow_measurement=np.array(result["metadata"]["ramp_flow_measurement"]),
                 ramp_model=np.array(result["metadata"]["ramp_model"]), ramp_ref_vph=np.array(result["metadata"]["ramp_ref_vph"]),
                 ramp_discharge_vph=np.array(result["metadata"]["ramp_discharge_vph"]), seed=np.array(sim_config["simulation"]["seed"]),
                 mainline_demand_vph=np.array(demand_vph), ramp_demand_vph=np.array(ramp_vph), teleports=np.array(result["metadata"]["teleports"]),
                 family=np.array(spec.name), family_id=np.array(spec.family_id), local_index=np.array(spec.local_index))
        saved.append(path)
        print(f"  [{g:>5}/{len(plan)}] {spec.name:<20s} #{spec.local_index:05d} demand={demand_vph:.0f}+{ramp_vph:.0f} "
              f"teleports={result['metadata']['teleports']}", flush=True)
    return saved


# ── Train/val/test splits ────────────────────────────────────────────────────


def make_splits(
    raw_dir: str,
    splits_dir: str,
    config: dict,
    seed: int = 42,
    stratify_by_family: bool = False,
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

    # Find all base samples (not truncated): sim_XXXX.npz or <family>_XXXXX.npz
    npz_files = sorted(raw_path.glob("sim_[0-9][0-9][0-9][0-9].npz")) or sorted(
        p for p in raw_path.glob("*.npz") if p.stem.rsplit("_", 1)[-1].isdigit()
    )
    if not npz_files:
        raise FileNotFoundError(f"No sim_*.npz files found in {raw_dir}")

    n = len(npz_files)
    rng = np.random.default_rng(seed)
    train_frac = config["train_frac"]
    val_frac = config["val_frac"]
    splits = {"train": [], "val": [], "test": []}
    groups = {}
    for i, p in enumerate(npz_files):
        key = p.stem.rsplit("_", 1)[0] if stratify_by_family else "all"
        groups.setdefault(key, []).append(i)
    for key, idx in groups.items():
        idx = list(rng.permutation(idx))
        n_g = len(idx)
        n_train = int(n_g * train_frac)
        n_val = int(n_g * (train_frac + val_frac)) - n_train
        splits["train"] += [npz_files[i].name for i in idx[:n_train]]
        splits["val"] += [npz_files[i].name for i in idx[n_train:n_train + n_val]]
        splits["test"] += [npz_files[i].name for i in idx[n_train + n_val:]]

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
        help="Override n_samples from config",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Start at one plus the largest existing sim_*.npz index in "
            "the configured output.raw_dir."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start writing at this sample index, for example 120.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing sim_*.npz files.",
    )
    parser.add_argument("--reuse-network", action="store_true", help="family schema: reuse a pre-built network dir")
    parser.add_argument("--no-splits", action="store_true", help="family schema: skip make_splits (the launcher does it once)")
    parser.add_argument("--indices-file", default=None, help="family schema: JSON list of global plan indices to generate")
    args = parser.parse_args()
    if args.append and args.start_index is not None:
        parser.error("Use either --append or --start-index, not both.")

    config_path = str(_PROJECT_ROOT / args.config)
    ds_cfg = load_config(config_path)

    if "families" in ds_cfg["dataset"]:
        indices = json.loads(Path(args.indices_file).read_text()) if args.indices_file else None
        generate_family_dataset(ds_cfg, _PROJECT_ROOT, indices=indices, reuse_network=args.reuse_network, overwrite=args.overwrite)
        if not args.no_splits:
            out = ds_cfg["output"]
            make_splits(str(_PROJECT_ROOT / out["raw_dir"]), str(_PROJECT_ROOT / out["splits_dir"]), ds_cfg["splits"],
                        seed=int(ds_cfg["dataset"].get("base_seed", 42)), stratify_by_family=True)
        sys.exit(0)

    # Allow CLI override of n_samples for quick smoke tests
    if args.n_samples is not None:
        ds_cfg["dataset"]["n_samples"] = args.n_samples
    if args.append:
        raw_dir = _PROJECT_ROOT / ds_cfg["output"]["raw_dir"]
        ds_cfg["dataset"]["start_index"] = next_sample_index(raw_dir)
    elif args.start_index is not None:
        ds_cfg["dataset"]["start_index"] = args.start_index
    if args.overwrite:
        ds_cfg["dataset"]["overwrite"] = True

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
