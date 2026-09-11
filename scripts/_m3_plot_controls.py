"""Plot the ramp-control input u(t) for each of the 4 spot-check rollouts."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_234849" / "eval_m3_spotcheck_1500vph"
DATA_DIR = _ROOT / "data" / "raw" / "m3_check_1500vph"

ROLLOUTS = [
    ("constant",           "sim_m3chk_500_0000.npz",       "Constant (u≡0.5)"),
    ("piecewise_constant", "sim_m3chk_pwc_0000.npz",       "Piecewise-constant (random jumps)"),
    ("smooth",             "sim_m3chk_smooth_0000.npz",    "Smooth (linear interp through 4–8 random knots)"),
    ("ramp_step",          "sim_m3chk_rampstep_0000.npz",  "Ramp-step (linear ramp with flat shoulders)"),
]


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for family, fname, title_suffix in ROLLOUTS:
        d = np.load(str(DATA_DIR / fname), allow_pickle=True)
        u = d["ramp_control"].astype(np.float32)
        t = d["t_grid"].astype(np.float32)
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(t, u, lw=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("u(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{family} ramp control — {title_suffix}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = PLOT_DIR / f"{family}_control.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved: {out.relative_to(_ROOT)}  (u: min={u.min():.3f}, max={u.max():.3f}, mean={u.mean():.3f})")


if __name__ == "__main__":
    main()
