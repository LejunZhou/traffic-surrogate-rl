"""Plot the compiled v3b SUMO network (true netconvert geometry) with the E1 loops and
the meter insertion point. Needs data/raw/network_v3b/net.net.xml, built by
sumo_env.network_builder.build_network(configs/sumo/scenario_v3b.yaml).
Usage: python scripts/plot_v3b_sumo_network.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sumolib  # noqa: E402
from sumo_env.detectors import _detector_edge_at_index  # noqa: E402
from utils.config import load_config  # noqa: E402

NET = "data/raw/network_v3b/net.net.xml"
OUT = "_progress/figures/m14_v3b_sumo_network.png"
COL = {"highway_pre": "#3b4a5a", "highway_accel": "#3b4a5a", "highway_post": "#3b4a5a", "ramp": "#c2410c"}
INTERNAL = "#9ca3af"
DET = "#2563eb"
SIG = "#16a34a"


def main() -> None:
    cfg = load_config("configs/sumo/scenario_v3b.yaml")
    net = sumolib.net.readNet(NET, withInternal=True)
    dets = []
    for i in range(cfg["detectors"]["n_detectors"]):
        eid, pos, nl = _detector_edge_at_index(cfg, i)
        for ln in range(nl):
            lane = net.getLane(f"{eid}_{ln}")
            x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), pos)
            dets.append((i, ln, x, y))
    y0 = net.getNode("upstream").getCoord()[1]

    def draw(ax, lw):
        for e in net.getEdges(withInternal=True):
            c = COL.get(e.getID(), INTERNAL)
            for lane in e.getLanes():
                xs, ys = zip(*lane.getShape())
                ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="butt", zorder=2 if c != INTERNAL else 3)
        for _, _, x, y in dets:
            ax.plot(x, y, marker="|", ms=12, mew=1.6, color=DET, zorder=4)
        for n in net.getNodes():
            x, y = n.getCoord()
            ax.plot(x, y, "o", ms=4, color="k", zorder=5)
            ax.annotate(f"{n.getID()} ({n.getType()})", (x, y), (0, 10), textcoords="offset points",
                        fontsize=7.5, ha="center")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.8), gridspec_kw=dict(height_ratios=[1, 1.5]))
    ax = axes[0]
    draw(ax, 4)
    ax.set_title("v3b compiled SUMO network (data/raw/network_v3b/net.net.xml), netconvert coordinates, "
                 "equal axes; blue ticks = 19 E1 loop stations", loc="left", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-40, 2040); ax.set_ylim(y0 - 130, y0 + 40)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    ax = axes[1]
    draw(ax, 7)
    ax.set_title("Merge area: ramp 191 m at 33.33 m/s, internal merge link 3.8 m capped at 9.18 m/s by netconvert, "
                 "2-lane accel edge 99 m, zipper drop\n"
                 "measured with an empty mainline: 71 km/h max on the ramp, 33 km/h at the nose, "
                 "81 km/h at the lane drop", loc="left", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(1090, 1560); ax.set_ylim(y0 - 118, y0 + 30)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    rs = net.getNode("ramp_start").getCoord()
    ax.plot(*rs, marker="s", ms=8, color=SIG, zorder=6)
    ax.annotate("meter stop bar = insertion point on ramp_0\n"
                "(departPos 0, departSpeed 0, one veh per green, D = 1200 veh/h)",
                rs, (rs[0] + 130, rs[1] + 8), fontsize=8, color=SIG, ha="left",
                arrowprops=dict(arrowstyle="->", color=SIG))
    m = {ln: (x, y) for i, ln, x, y in dets if i == 12}
    ax.annotate("station 12 lane 0 (accel lane, det index 0): ignored in v3b", m[0],
                (m[0][0] + 60, m[0][1] - 40), fontsize=8, color=DET, arrowprops=dict(arrowstyle="->", color=DET))
    ax.annotate("station 12 lane 1 (through lane): density source", m[1],
                (m[1][0] + 60, m[1][1] + 18), fontsize=8, color=DET, arrowprops=dict(arrowstyle="->", color=DET))
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
