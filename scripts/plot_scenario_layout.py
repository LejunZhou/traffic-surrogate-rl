"""Schematic of the SUMO road configuration (M14): the v3 layout as implemented
(stop bar 100 m before the nose, 200 m of storage road upstream of it) and the
proposed v3b layout (simulated ramp = acceleration segment only, storage in the
virtual queue). Not to scale in y. Usage: python scripts/plot_scenario_layout.py
"""
from __future__ import annotations
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = "_progress/figures/m14_ramp_layout_options"
MAIN = "#3b4a5a"; RAMP = "#c2410c"; VIRT = "#6b7280"; DET = "#2563eb"; SIG = "#16a34a"
HW_LEN, NOSE, ACC = 2000.0, 1300.0, 100.0
ANGLE = math.radians(30)


def ramp_pts(start_m: float, end_m: float):
    """Points along the ramp axis; `d` = metres upstream of the nose → (x, y). The ramp is
    drawn steeper than 30° so that its segments and labels stay readable (schematic)."""
    def p(d):
        return NOSE - 0.9 * d, -0.30 - 0.0085 * d
    return p(start_m), p(end_m)


def draw_mainline(ax):
    lw = 9
    ax.plot([0, HW_LEN], [0, 0], color=MAIN, lw=lw, solid_capstyle="butt", zorder=2)
    ax.plot([NOSE, NOSE + ACC], [-0.28, -0.28], color=MAIN, lw=lw, solid_capstyle="butt", zorder=2)
    ax.add_patch(Polygon([[NOSE + ACC, -0.28 - 0.05], [NOSE + ACC, -0.28 + 0.05], [NOSE + ACC + 60, 0.0]],
                         closed=True, color=MAIN, zorder=2))
    for x in range(100, 2000, 100):
        ax.plot([x, x], [-0.12, 0.12], color=DET, lw=1.2, zorder=3)
    ax.text(50, 0.22, "mainline: 1 lane, 2000 m, 120 km/h, IDM (τ = 1.0 s, speedDev 0.03)", fontsize=9, color=MAIN)
    ax.text(100, -0.32, "19 loop stations every 100 m (density by occupancy)", fontsize=8, color=DET)
    ax.annotate("merge nose 1300 m", (NOSE, 0.05), (NOSE - 40, 0.62), fontsize=8, ha="right",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax.annotate("zipper lane drop 1400 m", (NOSE + ACC, -0.3), (NOSE + ACC + 130, -1.0), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax.text(NOSE + ACC + 80, -0.30, "acceleration lane 100 m", fontsize=8, ha="left", va="center", color=MAIN)
    ax.annotate("exit flow = vehicles arrived\nat the downstream end", (HW_LEN, 0.0), (HW_LEN - 60, 0.62),
                fontsize=8, ha="right", arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.annotate("mainline demand d_k\n(per 5-min block)", (0, 0.0), (40, -0.8), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))


def draw_ramp_segment(ax, d_from, d_to, color, label, lw=7):
    (x0, y0), (x1, y1) = ramp_pts(d_from, d_to)
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=2)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(xm + 45, ym + 0.02, label, fontsize=8, ha="left", va="center", color=color)


def draw_signal(ax, d, text):
    (x, y), _ = ramp_pts(d, d)
    ax.plot([x - 12, x + 12], [y + 0.12, y - 0.12], color=SIG, lw=3, zorder=4)  # stop bar across the ramp
    ax.plot(x - 28, y + 0.05, marker="o", color=SIG, ms=9, zorder=4)                # signal head
    ax.annotate(text, (x - 34, y + 0.05), (x - 90, y - 0.02), fontsize=8, ha="right", va="bottom", color=SIG,
                arrowprops=dict(arrowstyle="-", color=SIG, lw=0.8))


def draw_virtual_box(ax, d_anchor, text):
    (xa, ya), _ = ramp_pts(d_anchor, d_anchor)
    x0, y0, w, h = 120, ya - 0.95, 620, 0.75
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", fc="#f3f4f6", ec=VIRT, ls="--", lw=1.2, zorder=1))
    ax.text(x0 + w / 2, y0 + h / 2, text, fontsize=8, ha="center", va="center", color=VIRT)
    ax.annotate("", (xa - 10, ya - 0.06), (x0 + w, y0 + h * 0.75), arrowprops=dict(arrowstyle="->", color=VIRT, lw=1.0, ls="--"))
    ax.annotate("ramp demand r_k (per 5-min block)", (x0, y0 + h / 2), (x0 - 10, y0 + h / 2 - 0.55), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))


def panel_v3(ax):
    ax.set_title("(a) v3 as implemented: 300 m ramp, stop bar 100 m upstream of the nose", loc="left", fontsize=10)
    draw_mainline(ax)
    draw_ramp_segment(ax, 100, 0, RAMP, "ramp, 100 m at 60 km/h (acceleration run)")
    draw_ramp_segment(ax, 300, 100, "#f2b48e", "ramp, 200 m at 60 km/h\n(storage road: exists in the network,\nnever driven)")
    draw_signal(ax, 100, "stop bar 100 m before the nose:\nvehicles inserted here from standstill\nat u_k·D veh/h (D = 1600)")
    draw_virtual_box(ax, 300, "virtual queue Q_k (a counter)\nQ_k = Q_{k−1} + a_k − release_k\ncap 28 veh = 200 m storage road / 7 m")


def panel_v3b(ax):
    ax.set_title("(b) v3b: simulated ramp = 200 m acceleration segment (120 km/h), storage in the virtual queue, no cap",
                 loc="left", fontsize=10)
    draw_mainline(ax)
    draw_ramp_segment(ax, 200, 0, RAMP, "ramp, 200 m at 120 km/h (acceleration run)")
    draw_signal(ax, 200, "meter signal at the start of the simulated ramp:\none vehicle per green, starting from 0 km/h\nrelease_k = min(Q_k + a_k, u_k·D·Δt/3600), D = 1200 veh/h")
    draw_virtual_box(ax, 200, "virtual storage Q_k (not simulated, uncapped)\nQ_k = Q_{k−1} + a_k − release_k\nwaiting time Q_k·Δt is charged in the TTS reward")


def panel_final(ax):
    """v3b as configured (configs/sumo/scenario_v3b.yaml): the study scenario."""
    ax.set_title("Scenario v3b (final): 2000 m single-lane mainline, metered on-ramp at 1300 m, virtual ramp storage",
                 loc="left", fontsize=10)
    draw_mainline(ax)
    draw_ramp_segment(ax, 200, 0, RAMP, "ramp, 200 m at 120 km/h (acceleration run,\n≈ 100 km/h at the nose; + 100 m acceleration lane)")
    draw_signal(ax, 200, "meter signal (stop bar) at the start of the simulated ramp:\none vehicle per green, inserted from 0 km/h\nrelease_k = min(Q_k + a_k, u_k·D·Δt/3600), D = 1200 veh/h\nu_k ∈ [0, 1] = the policy's action, held for Δt = 30 s")
    draw_virtual_box(ax, 200, "virtual storage Q_k (not simulated, no cap)\nQ_k = Q_{k−1} + a_k − release_k\ncharged in the TTS reward every step (Q_k·Δt);\nno end-of-hour charge (terminal_queue_weight 0)")
    # merge station detail: two loops at 1300 m, through-lane density only
    ax.plot([NOSE, NOSE], [-0.28 - 0.12, -0.28 + 0.12], color=DET, lw=1.2, zorder=3)
    ax.annotate("station 12 at the merge nose has two loops;\ndensity = through lane only (v3b),\nthe acceleration-lane loop is ignored",
                (NOSE + 5, -0.40), (NOSE + 60, -2.25), fontsize=8, ha="left", color=DET,
                arrowprops=dict(arrowstyle="->", color=DET, lw=0.8))
    ax.text(NOSE + 60, -2.75, "density per station: occupancy × 1000 / 5 m, clipped at 143 veh/km\nTTS_k = (N_k + Q_k + P_k)·Δt = (offered − served)·Δt by conservation",
            fontsize=8, ha="left", va="top", color="0.25")


def main_final():
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.4))
    panel_final(ax)
    ax.set_xlim(-120, HW_LEN + 80); ax.set_ylim(-3.3, 1.1)
    ax.set_yticks([]); ax.set_xlabel("position along the mainline [m]", fontsize=9)
    for sp in ("left", "right", "top"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = "_progress/figures/m14_v3b_road_layout"
    fig.savefig(out + ".png", dpi=170); fig.savefig(out + ".pdf")
    print("wrote", out + ".png")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.0), gridspec_kw=dict(height_ratios=[4.9, 3.3]))
    for ax, fn, ylo in zip(axes, (panel_v3, panel_v3b), (-4.0, -3.3)):
        fn(ax)
        ax.set_xlim(-120, HW_LEN + 80); ax.set_ylim(ylo, 0.9)
        ax.set_yticks([]); ax.set_xlabel("position along the mainline [m]", fontsize=9)
        for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    fig.suptitle("SUMO road configuration: metered on-ramp with virtual storage (schematic, y not to scale)",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT + ".png", dpi=170); fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    import sys
    main_final() if "--final" in sys.argv else main()
