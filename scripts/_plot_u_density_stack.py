"""Stacked u(t) + density-heatmap figures for the 2000-vph inspection rollouts.

Top panel: ramp metering control u(t) (piecewise-constant, steps-post).
Bottom panel: SUMO density space-time heatmap (same hot_r style as
utils.plotting.plot_trajectory), sharing the time axis.

Panels are kept the same width despite the heatmap colorbar by appending a
matching (invisible) colorbar slot to the top axis via make_axes_locatable.

Writes sim_XXXX_u_density.png next to the originals; does not touch them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

_ROOT = Path(__file__).resolve().parent.parent
INSPECT_DIR = _ROOT / "data" / "raw" / "inspect_2000vph_5rollouts"

# Control family per sample index (M2 round-robin over 4 families, 5 samples).
FAMILIES = ["constant", "piecewise_constant", "smooth", "ramp_step", "constant"]


def plot_one(npz_path: Path, family: str) -> Path:
    z = np.load(npz_path)
    density = z["density"]            # (N_x, T_ctrl)
    x_grid = z["x_grid"].astype(float)
    t_grid = z["t_grid"].astype(float)
    u = z["ramp_control"].astype(float)
    demand = float(z["mainline_demand_vph"])
    seed = int(z["seed"])

    t_min, t_max = float(t_grid[0]), float(t_grid[-1])
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])

    fig, (ax_u, ax_rho) = plt.subplots(
        2, 1, figsize=(10, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
    )

    # --- Top: control signal u(t) ---
    ax_u.plot(t_grid, u, color="tab:blue", lw=1.6, drawstyle="steps-post")
    ax_u.fill_between(t_grid, 0, u, step="post", color="tab:blue", alpha=0.15)
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_ylabel("u(t)", fontsize=11)
    ax_u.grid(alpha=0.3)
    ax_u.set_title(
        f"{npz_path.stem} — {int(demand)} vph, {family}, seed={seed} "
        f"(u_mean={u.mean():.2f})",
        fontsize=12,
    )
    # Invisible colorbar slot so the top panel width matches the heatmap below.
    cax_u = make_axes_locatable(ax_u).append_axes("right", size="3%", pad=0.1)
    cax_u.set_axis_off()

    # --- Bottom: density heatmap ---
    im = ax_rho.imshow(
        density, aspect="auto", origin="lower",
        extent=[t_min, t_max, x_min, x_max],
        cmap="hot_r", interpolation="nearest",
    )
    ax_rho.set_xlim(t_min, t_max)
    ax_rho.set_xlabel("Time [s]", fontsize=11)
    ax_rho.set_ylabel("Position [m]", fontsize=11)
    # Mark the on-ramp merge location.
    ax_rho.axhline(1300.0, color="cyan", lw=1.0, ls="--", alpha=0.7)

    cax_rho = make_axes_locatable(ax_rho).append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax_rho)
    cbar.set_label("Density [veh/km]", fontsize=11)

    out_path = npz_path.with_name(npz_path.stem + "_u_density.png")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    npz_files = sorted(INSPECT_DIR.glob("sim_[0-9][0-9][0-9][0-9].npz"))
    if not npz_files:
        sys.exit(f"No sim_*.npz files found in {INSPECT_DIR}")
    for i, npz_path in enumerate(npz_files):
        family = FAMILIES[i] if i < len(FAMILIES) else "?"
        out = plot_one(npz_path, family)
        print(f"saved: {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
