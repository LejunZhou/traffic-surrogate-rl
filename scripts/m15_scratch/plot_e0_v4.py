"""Figure for the E0 characterisation report (M15 §4): per-profile returns of the constant
grid, the best hand schedule and the single best constant, storage-mandatory profiles marked.
    python scripts/m15_scratch/plot_e0_v4.py [--report _progress/m15_e0_v4_characterisation.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--report", default=str(ROOT / "_progress/m15_e0_v4_characterisation.json"))
ap.add_argument("--out", default=str(ROOT / "_progress/figures/m15/e0_v4.png"))
args = ap.parse_args()
rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
gate = rep["gate"]; cap = rep["capacity"]; fl = rep["flush"]
keys = sorted(gate, key=lambda k: gate[k]["peak_merge_load"] if "peak_merge_load" in gate[k] else gate[k]["peak_total"])
n = len(keys); us = sorted(next(iter(cap.values())), key=float)
u_single = rep["gate_summary"]["single_constant_u"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
x = np.arange(n)
for j, k in enumerate(keys):
    g = gate[k]
    rc = np.array([cap[k][u]["return"] for u in us]); bd = np.array([cap[k][u]["breakdown"] for u in us])
    ax.plot(np.full(len(us), j) + np.linspace(-0.28, 0.28, len(us)), rc, ".", color="#9ca3af", ms=6, zorder=2)
    ax.plot(np.full(int(bd.sum()), j) + np.linspace(-0.28, 0.28, len(us))[bd], rc[bd], "x", color="#9ca3af", ms=5, zorder=2)
    ax.plot(j, g["best_constant_return"], "o", color="#2563eb", ms=7, zorder=4)
    ax.plot(j, g["single_constant_return"], "s", color="#7c3aed", ms=6, zorder=4)
    ax.plot(j, g["best_flush_return"], "D", color="#c2410c", ms=7, zorder=5)
    if g["storage_needed"]:
        ax.axvspan(j - 0.45, j + 0.45, color="#fde68a", alpha=0.35, zorder=0)
ax.plot([], [], ".", color="#9ca3af", label="constant u grid (x = breakdown)")
ax.plot([], [], "o", color="#2563eb", label="best constant (per profile)")
ax.plot([], [], "s", color="#7c3aed", label=f"single best constant u = {u_single}")
ax.plot([], [], "D", color="#c2410c", label="best hand schedule")
ax.axvspan(-1, -0.9, color="#fde68a", alpha=0.35, label="storage-mandatory (lane-0 load > threshold)")
ax.set_xlim(-0.6, n - 0.4)
ax.set_xticks(x); ax.set_xticklabels([f"{gate[k]['profile'].split(']')[0].split('[')[1]}\n{gate[k].get('peak_merge_load', gate[k]['peak_total']):.0f}" for k in keys], fontsize=8)
ax.set_xlabel("E0 profile index / peak lane-0 load [veh/h]"); ax.set_ylabel("return (−TTS, veh h)")
ax.set_title("E0 on v4: returns per profile", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(alpha=0.25, axis="y")

ax = axes[1]
m_best = np.array([gate[k]["margin"] for k in keys]); m_single = np.array([gate[k]["margin_single"] for k in keys])
sm = np.array([gate[k]["storage_needed"] for k in keys])
ax.bar(x - 0.18, m_best, 0.36, color="#2563eb", label="best schedule − best constant")
ax.bar(x + 0.18, m_single, 0.36, color="#7c3aed", label=f"best schedule − single constant u = {u_single}")
ax.axhline(rep["gate_summary"]["margin_required"], color="#111827", lw=1, ls="--"); ax.text(n - 0.5, rep["gate_summary"]["margin_required"] + 1, "gate margin", ha="right", fontsize=8.5)
ax.axhline(0, color="#6b7280", lw=0.8)
for j in np.where(sm)[0]:
    ax.axvspan(j - 0.45, j + 0.45, color="#fde68a", alpha=0.35, zorder=0)
ax.set_xticks(x); ax.set_xticklabels([gate[k]["profile"].split("]")[0].split("[")[1] for k in keys], fontsize=8)
ax.set_xlabel("E0 profile index"); ax.set_ylabel("margin [veh h]")
gs = rep["gate_summary"]
ax.set_title(f"margin over the constants (storage-mandatory pass: {gs['n_storage_passed']}/{gs['n_storage_needed']} per-profile, "
             f"{gs['n_storage_passed_single']}/{gs['n_storage_needed']} single)", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y")
for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print("wrote", args.out)
