"""Figure for M15 progress §3: exit flow vs offered flow per ramp level (capacity sweep on v4).
    python scripts/m15_scratch/plot_capacity_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
rep = json.loads((ROOT / "_progress/m15_capacity_sweep_v4.json").read_text(encoding="utf-8"))
rows = rep["rows"]
COL = {0: "#6b7280", 400: "#2563eb", 700: "#c2410c", 800: "#7c3aed"}     # fixed hue per ramp level
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for r in sorted(COL):
    rr = [w for w in rows if w["ramp"] == r]
    if not rr:
        continue
    off = np.array([w["offered"] for w in rr]); ex = np.array([w["exit_vph_last30"] for w in rr]); bd = np.array([w["breakdown"] for w in rr])
    ax.plot(off, ex, "o-", color=COL[r], lw=2, ms=6, label=f"ramp {r} veh/h", zorder=3)
    ax.plot(off[bd], ex[bd], "x", color=COL[r], ms=11, mew=2.2, zorder=4)
ax.plot([5400, 6700], [5400, 6700], ":", color="#9ca3af", lw=1.2, label="exit = offered")
ax.set_xlabel("offered flow, mainline + ramp [veh/h]"); ax.set_ylabel("exit flow, last 30 min [veh/h]")
ax.set_title("v4 merge: throughput vs offered flow (x = breakdown flagged, threshold 30)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
ax = axes[1]
for r in sorted(COL):
    rr = [w for w in rows if w["ramp"] == r]
    if not rr:
        continue
    load = np.array([w["mainline"] / 3 + w["ramp"] for w in rr]); bd = np.array([w["breakdown"] for w in rr])
    rho = np.array([w["rho_max_mean_lane"] for w in rr])
    ax.plot(load, rho, "o", color=COL[r], ms=7, label=f"ramp {r} veh/h", zorder=3)
    ax.plot(load[bd], rho[bd], "x", color=COL[r], ms=11, mew=2.2, zorder=4)
ax.axvline(2450, color="#111827", lw=1, ls="--"); ax.text(2455, 24, "lane-0 load 2450", fontsize=8.5, rotation=90, va="bottom")
ax.axhline(30, color="#9ca3af", lw=1, ls=":"); ax.text(1950, 30.5, "breakdown threshold 30 (lane mean)", fontsize=8.5, color="#6b7280")
ax.set_xlabel("lane-0 load, mainline / 3 + ramp [veh/h]"); ax.set_ylabel("max through-lane-mean density [veh/km]")
ax.set_title("breakdown is a lane-0 load event: d/3 + r above ~2450 veh/h", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
out = ROOT / "_progress/figures/m15/capacity_sweep_v4.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
