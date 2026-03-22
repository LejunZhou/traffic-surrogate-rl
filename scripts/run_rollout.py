"""
Milestone 1 entry point: run one SUMO simulation rollout and save results.

Usage (from project root):
    python scripts/run_rollout.py [--config configs/sumo/phase1.yaml]
                                  [--ramp-rate 0.5]
                                  [--output-index 0000]

What this script does:
1. Loads the Phase 1 SUMO config.
2. Builds network files (nodes, edges, net.xml, routes.rou.xml) in
   <output.network_dir> using netconvert.
3. Builds the detector additional file (.add.xml).
4. Runs one SUMO simulation with a constant ramp metering rate.
5. Saves the trajectory to data/raw/sim_<index>.npz.
6. Saves a diagnostic density heatmap to data/raw/sim_<index>_density.png.
7. Prints a brief summary.

Prerequisites:
  - SUMO ≥ 1.18 installed and on PATH (sumo, netconvert)
  - SUMO Python tools on PYTHONPATH:
      export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
  - Project installed in editable mode:
      pip install -e .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the project root without installing
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import load_config, merge_configs
from sumo_env.network_builder import build_network
from sumo_env.detectors import build_detector_file
from sumo_env.run_simulation import run_simulation
from utils.plotting import plot_trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one SUMO rollout (Milestone 1)")
    p.add_argument(
        "--config",
        default="configs/sumo/phase1.yaml",
        help="Path to YAML config (default: configs/sumo/phase1.yaml)",
    )
    p.add_argument(
        "--ramp-rate",
        type=float,
        default=0.5,
        help="Constant ramp metering rate ∈ [0, 1] for this rollout (default: 0.5)",
    )
    p.add_argument(
        "--output-index",
        default="0000",
        help="Zero-padded index used in output filenames (default: 0000)",
    )
    p.add_argument(
        "--set",
        action="append",
        metavar="SECTION.KEY=VALUE",
        default=[],
        dest="overrides",
        help=(
            "Override a config value using dot-notation key=value pairs "
            "(can be repeated), e.g. --set demand.mainline_demand_vph=1000 "
            "--set simulation.seed=7"
        ),
    )
    return p.parse_args()


def _parse_set_overrides(items: list[str]) -> dict:
    """Parse --set SECTION.KEY=VALUE strings into a nested override dict.

    Supports one level of nesting (SECTION.KEY).  Values are coerced to int,
    then float, then left as str.

    Args:
        items: List of strings like ["demand.mainline_demand_vph=1200", ...].

    Returns:
        Nested dict suitable for passing to merge_configs().

    Raises:
        ValueError: If an item does not contain '='.
    """
    result: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--set override must be in KEY=VALUE format, got: {item!r}"
            )
        key, raw_val = item.split("=", 1)
        parts = key.split(".")

        # Coerce value: int → float → str
        val: int | float | str
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                val = raw_val

        # Build nested dict one level at a time
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val

    return result


def main() -> None:
    args = parse_args()

    # ── 1. Load config (with optional --set overrides) ────────────────────────
    config_path = _PROJECT_ROOT / args.config
    config = load_config(str(config_path))

    if args.overrides:
        override_dict = _parse_set_overrides(args.overrides)
        config = merge_configs(config, override_dict)
        print(f"[run_rollout] Overrides    : {args.overrides}")

    sim_cfg = config["simulation"]
    out_cfg = config["output"]

    T_ctrl = int(sim_cfg["duration_s"] / sim_cfg["dt_ctrl_s"])  # 120
    ramp_control = np.full(T_ctrl, args.ramp_rate, dtype=np.float32)

    warmup_s = sim_cfg.get("ramp_warmup_s", 0.0)
    print(f"[run_rollout] Config      : {config_path}")
    print(f"[run_rollout] Ramp rate   : {args.ramp_rate:.2f} (constant throughout)")
    print(f"[run_rollout] T_ctrl      : {T_ctrl} steps × {sim_cfg['dt_ctrl_s']} s")
    print(f"[run_rollout] Ramp warmup : {warmup_s:.0f} s "
          f"({int(warmup_s / sim_cfg['dt_ctrl_s'])} control steps with no ramp insertion)")

    # ── 2. Build network ──────────────────────────────────────────────────────
    network_dir = _PROJECT_ROOT / out_cfg["network_dir"]
    print(f"[run_rollout] Building network in {network_dir} …")
    network_files = build_network(str(network_dir), config)
    print(f"[run_rollout] Network     : {network_files['net']}")
    print(f"[run_rollout] Routes      : {network_files['route']}")

    # ── 3. Build detector file ────────────────────────────────────────────────
    det_file = str(network_dir / "detectors.add.xml")
    build_detector_file(det_file, config)
    print(f"[run_rollout] Detectors   : {det_file}")

    # ── 4. Run simulation ─────────────────────────────────────────────────────
    print(f"[run_rollout] Starting SUMO ({sim_cfg['sumo_binary']}) …")
    result = run_simulation(
        net_file=network_files["net"],
        route_file=network_files["route"],
        detector_file=det_file,
        ramp_control=ramp_control,
        config=config,
    )
    print("[run_rollout] Simulation complete.")

    # ── 5. Save trajectory ────────────────────────────────────────────────────
    raw_dir = _PROJECT_ROOT / out_cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    traj_path = raw_dir / f"sim_{args.output_index}.npz"

    np.savez(
        str(traj_path),
        density=result["density"],
        speed=result["speed"],
        flow=result["flow"],
        x_grid=result["x_grid"],
        t_grid=result["t_grid"],
        mainline_demand=result["mainline_demand"],
        ramp_control=result["ramp_control"],
        # scalar metadata stored as 0-d arrays
        seed=np.array(result["metadata"]["seed"]),
        mainline_demand_vph=np.array(result["metadata"]["mainline_demand_vph"]),
        ramp_demand_vph=np.array(result["metadata"]["ramp_demand_vph"]),
    )
    print(f"[run_rollout] Trajectory  → {traj_path}")

    # ── 6. Diagnostic plot ────────────────────────────────────────────────────
    plot_path = raw_dir / f"sim_{args.output_index}_density.png"
    plot_trajectory(
        density=result["density"],
        x_grid=result["x_grid"],
        t_grid=result["t_grid"],
        output_path=plot_path,
        title=(
            f"Density heatmap — {config['demand']['demand_profile']}, "
            f"ramp rate={args.ramp_rate:.2f}, seed={sim_cfg['seed']}"
        ),
    )
    print(f"[run_rollout] Density plot → {plot_path}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    d = result["density"]
    meta = result["metadata"]
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  density  : mean={d.mean():.2f}, max={d.max():.2f}  veh/km")
    print(f"  speed    : mean={result['speed'].mean():.1f}  km/h")
    print(f"  flow     : mean={result['flow'].mean():.0f}  veh/hr")
    print(f"  shape    : density{d.shape}  (N_x × T_ctrl)")
    print(f"  insertions: {meta['insert_success']}/{meta['insert_attempts']} "
          f"ok, {meta['insert_rejected']} rejected")
    print(f"  teleports : {meta['teleports']}")
    print(f"  ramp queue: max={meta.get('ramp_queue_max', '?')}, "
          f"mean={meta.get('ramp_queue_mean', '?'):.1f}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
