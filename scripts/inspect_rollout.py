"""
Inspect a saved simulation rollout (.npz) and print a compact summary.

Usage (from project root):
    python scripts/inspect_rollout.py data/raw/sim_0000.npz
    python scripts/inspect_rollout.py data/raw/sim_0000.npz --full

What this prints:
1. Keys and array shapes stored in the .npz file.
2. Per-detector statistics for density: mean, std, min, max across all time steps.
3. Per-time-step statistics for density: mean, std, min, max across all detectors.
4. Scalar metadata (if present).
5. With --full: also prints the raw density matrix (one row per detector).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a saved SUMO rollout .npz file")
    p.add_argument("npz_path", help="Path to the .npz file to inspect")
    p.add_argument(
        "--full",
        action="store_true",
        help="Also print the full density matrix (N_x rows × T_ctrl columns)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.npz_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(str(path), allow_pickle=False)

    # ── 1. Keys and shapes ────────────────────────────────────────────────────
    print(f"\nFile: {path}")
    print(f"{'─' * 60}")
    print("Keys and shapes:")
    for key in sorted(data.files):
        arr = data[key]
        print(f"  {key:<25s}  shape={arr.shape}  dtype={arr.dtype}")

    # ── 2. Scalar metadata ────────────────────────────────────────────────────
    scalar_keys = [k for k in data.files if data[k].ndim == 0]
    if scalar_keys:
        print(f"\nScalar metadata:")
        for key in sorted(scalar_keys):
            print(f"  {key:<25s}  = {data[key].item()}")

    # ── 3. Per-detector density stats ─────────────────────────────────────────
    if "density" not in data.files:
        print("\nNo 'density' array found — nothing more to show.")
        return

    density = data["density"]   # (N_x, T_ctrl)
    N_x, T_ctrl = density.shape

    x_grid = data["x_grid"] if "x_grid" in data.files else np.arange(N_x, dtype=float)

    print(f"\nPer-detector density stats  [veh/km]  (N_x={N_x}, T_ctrl={T_ctrl}):")
    header = f"  {'det':>4}  {'pos_m':>7}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}"
    print(header)
    print(f"  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for i in range(N_x):
        row = density[i, :]
        print(
            f"  {i:>4d}  {x_grid[i]:>7.0f}  "
            f"{row.mean():>8.2f}  {row.std():>8.2f}  "
            f"{row.min():>8.2f}  {row.max():>8.2f}"
        )

    # ── 4. Per-time-step density stats ────────────────────────────────────────
    t_grid = data["t_grid"] if "t_grid" in data.files else np.arange(T_ctrl, dtype=float)

    print(f"\nPer-time-step density stats  [veh/km]  (showing every 10th step):")
    header = f"  {'step':>5}  {'t_s':>6}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}"
    print(header)
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for k in range(0, T_ctrl, max(1, T_ctrl // 12)):
        col = density[:, k]
        print(
            f"  {k:>5d}  {t_grid[k]:>6.0f}  "
            f"{col.mean():>8.2f}  {col.std():>8.2f}  "
            f"{col.min():>8.2f}  {col.max():>8.2f}"
        )

    # Overall summary
    print(f"\nOverall density: mean={density.mean():.2f}, "
          f"std={density.std():.2f}, "
          f"min={density.min():.2f}, "
          f"max={density.max():.2f}  veh/km")

    if "speed" in data.files:
        speed = data["speed"]
        print(f"Overall speed  : mean={speed.mean():.1f}, "
              f"min={speed.min():.1f}, "
              f"max={speed.max():.1f}  km/h")

    if "flow" in data.files:
        flow = data["flow"]
        print(f"Overall flow   : mean={flow.mean():.0f}, "
              f"min={flow.min():.0f}, "
              f"max={flow.max():.0f}  veh/hr")

    # ── 5. Full density matrix (optional) ────────────────────────────────────
    if args.full:
        print(f"\nFull density matrix  (N_x={N_x} rows × T_ctrl={T_ctrl} cols)  [veh/km]:")
        col_headers = "  ".join(f"{k:>6}" for k in range(T_ctrl))
        print(f"  {'det':>4}  {col_headers}")
        for i in range(N_x):
            vals = "  ".join(f"{v:>6.1f}" for v in density[i, :])
            print(f"  {i:>4d}  {vals}")

    print()


if __name__ == "__main__":
    main()
