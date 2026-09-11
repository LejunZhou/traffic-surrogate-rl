"""Macro + micro diagnostics for the 2000-vph on-ramp merge bottleneck.

Answers "why is density downstream of the 1400 m merge always low?":
  MACRO  -- loads sim_0000.npz, prints the across-merge profile, and plots
            the fundamental-diagram split (congested upstream branch vs
            free-flow downstream branch) + the spatial rho/v profile.
  MICRO  -- replays the constant u=0.77 metering via TraCI (mirrors
            run_simulation's insertion), freezes at SNAP_T, and reads every
            vehicle's true position/speed -> TRUE spatial density (count /
            length, immune to the detector q/v artifact) + per-100 m profile.

Figures are written into the inspection dir; the script is re-runnable.

    PYTHONPATH=src python scripts/_bottleneck_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
INSPECT_DIR = _ROOT / "data" / "raw" / "inspect_2000vph_5rollouts"
NETDIR = INSPECT_DIR / "network"
NPZ = INSPECT_DIR / "sim_0000.npz"

RAMP_POS, ACCEL_END = 1300.0, 1400.0
U = 0.77                  # constant metering rate (matches sim_0000)
RAMP_DEMAND_VPH = 800.0
STEP = 1.0
SNAP_T = 2400.0           # snapshot time (congestion well established)


# ── MACRO ────────────────────────────────────────────────────────────────────

def macro() -> None:
    z = np.load(NPZ)
    rho, spd, flw = z["density"], z["speed"], z["flow"]
    x, t, u = z["x_grid"], z["t_grid"], z["ramp_control"]
    m = t > 600  # established-congestion window

    print(f"MACRO  sim_0000: constant u={u.mean():.2f}, mainline 2000 + "
          f"ramp ~{u.mean()*800:.0f} vph; merge at x={RAMP_POS:.0f} m\n")
    print(f"{'x[m]':>6}{'region':>14}{'rho[veh/km]':>12}{'v[km/h]':>9}{'q[veh/h]':>10}")
    for i in range(len(x)):
        reg = ("pre (1ln)" if x[i] < RAMP_POS else
               "accel(2ln)" if x[i] < ACCEL_END else "post (1ln)")
        print(f"{x[i]:>6.0f}{reg:>14}{rho[i,m].mean():>12.1f}"
              f"{spd[i,m].mean():>9.1f}{flw[i,m].mean():>10.0f}")

    pre, post = x < RAMP_POS, x >= ACCEL_END
    print(f"\n  UPSTREAM  : rho={rho[pre][:,m].mean():6.1f}  v={spd[pre][:,m].mean():5.1f} km/h")
    print(f"  DOWNSTREAM: rho={rho[post][:,m].mean():6.1f}  v={spd[post][:,m].mean():5.1f} km/h"
          f"   (rho = q/v = {flw[post][:,m].mean():.0f}/{spd[post][:,m].mean():.0f} "
          f"= {flw[post][:,m].mean()/spd[post][:,m].mean():.0f})")

    # ── figure: FD split + spatial profile ──
    fig, (axfd, axsp) = plt.subplots(1, 2, figsize=(13, 4.5))

    region = np.where(x < RAMP_POS, 0, np.where(x < ACCEL_END, 1, 2))
    colors = {0: "tab:red", 1: "tab:orange", 2: "tab:blue"}
    labels = {0: "upstream (pre-merge)", 1: "merge/accel", 2: "downstream (post-merge)"}
    for r in (0, 1, 2):
        rows = np.where(region == r)[0]
        axfd.scatter(rho[rows][:, m].ravel(), flw[rows][:, m].ravel(),
                     s=8, alpha=0.25, color=colors[r], label=labels[r])
    axfd.set_xlabel("density [veh/km]"); axfd.set_ylabel("flow [veh/h]")
    axfd.set_title("Fundamental diagram: two branches\n(same flow, opposite density/speed)")
    axfd.legend(loc="upper right", fontsize=8); axfd.grid(alpha=0.3)

    axsp.plot(x, rho[:, m].mean(1), "o-", color="tab:red", label="density [veh/km]")
    axsp.set_xlabel("position x [m]"); axsp.set_ylabel("density [veh/km]", color="tab:red")
    axsp.tick_params(axis="y", labelcolor="tab:red")
    ax2 = axsp.twinx()
    ax2.plot(x, spd[:, m].mean(1), "s-", color="tab:blue", label="speed [km/h]")
    ax2.set_ylabel("speed [km/h]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    axsp.axvline(RAMP_POS, color="k", ls="--", lw=1, alpha=0.6)
    axsp.text(RAMP_POS + 15, axsp.get_ylim()[1]*0.9, "merge", fontsize=9)
    axsp.set_title("Across-merge profile (t > 600 s)")
    axsp.grid(alpha=0.3)

    fig.tight_layout()
    out = INSPECT_DIR / "bottleneck_macro_profile.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\n  saved: {out.relative_to(_ROOT)}")


# ── MICRO ────────────────────────────────────────────────────────────────────

def _abs_x(road, lanepos):
    if road == "highway_pre":   return lanepos
    if road == "highway_accel": return RAMP_POS + lanepos
    if road == "highway_post":  return ACCEL_END + lanepos
    return None


def micro() -> None:
    import traci
    cmd = ["sumo",
           "--net-file", str(NETDIR / "net.net.xml"),
           "--route-files", str(NETDIR / "routes_2000.rou.xml"),
           "--step-length", str(STEP), "--seed", "42",
           "--no-step-log", "--no-warnings", "--collision.action", "warn"]
    traci.start(cmd)
    rate, frac, vid = U * RAMP_DEMAND_VPH / 3600.0, 0.0, 0
    while traci.simulation.getTime() < SNAP_T:
        frac += rate * STEP
        n = int(frac); frac -= n
        for _ in range(n):
            try:
                traci.vehicle.add(vehID=f"ramp_{vid}", routeID="route_ramp",
                                  typeID="passenger", depart=str(traci.simulation.getTime()),
                                  departLane="first", departPos="free", departSpeed="0")
                vid += 1
            except traci.exceptions.TraCIException:
                pass
        traci.simulationStep()
    recs = []
    for v in traci.vehicle.getIDList():
        road = traci.vehicle.getRoadID(v)
        xx = _abs_x(road, traci.vehicle.getLanePosition(v))
        recs.append((road, xx, traci.vehicle.getSpeed(v) * 3.6))
    traci.close()

    hw = [r for r in recs if r[1] is not None]
    xs = np.array([r[1] for r in hw]); vs = np.array([r[2] for r in hw])
    pre = xs < RAMP_POS; post = xs >= ACCEL_END
    print(f"\nMICRO  snapshot t={SNAP_T:.0f}s, u={U}: {len(recs)} vehicles "
          f"({(np.array([r[0] for r in recs])=='ramp').sum()} queued on ramp)")
    print(f"  pre-merge : {pre.sum():3d} veh over 1200 m -> {pre.sum()/1.2:5.0f} veh/km/ln  "
          f"v={vs[pre].mean():5.1f} km/h")
    print(f"  post-merge: {post.sum():3d} veh over  600 m -> {post.sum()/0.6:5.0f} veh/km/ln  "
          f"v={vs[post].mean():5.1f} km/h")
    xp = np.sort(xs[pre]); xq = np.sort(xs[post])
    print(f"  spacing pre={np.diff(xp).mean():.1f} m (min {np.diff(xp).min():.1f}=jam)  "
          f"post={np.diff(xq).mean():.1f} m")

    # ── figure: per-100 m count+speed, and (x, speed) scatter ──
    fig, (axb, axsc) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 2], "hspace": 0.12})
    bins = np.arange(0, 2001, 100)
    cnt, _ = np.histogram(xs, bins=bins)
    centers = bins[:-1] + 50
    axb.bar(centers, cnt, width=90, color="tab:red", alpha=0.6, label="veh per 100 m")
    axb.set_ylabel("vehicles / 100 m", color="tab:red")
    axb.tick_params(axis="y", labelcolor="tab:red")
    ax2 = axb.twinx()
    mean_v = [vs[(xs >= b) & (xs < b+100)].mean() if ((xs >= b) & (xs < b+100)).any()
              else np.nan for b in bins[:-1]]
    ax2.plot(centers, mean_v, "o-", color="tab:blue", label="mean speed")
    ax2.set_ylabel("mean speed [km/h]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    axb.axvline(RAMP_POS, color="k", ls="--", lw=1, alpha=0.6)
    axb.set_title(f"Microscopic snapshot at t={SNAP_T:.0f} s (u={U}): "
                  f"jam upstream, free-flow downstream of the merge")

    sc = axsc.scatter(xs, vs, c=vs, cmap="RdYlGn", s=18, vmin=0, vmax=120)
    axsc.axvline(RAMP_POS, color="k", ls="--", lw=1, alpha=0.6)
    axsc.text(RAMP_POS + 15, 5, "merge", fontsize=9)
    axsc.set_xlabel("position x [m]"); axsc.set_ylabel("vehicle speed [km/h]")
    axsc.set_ylim(-5, 125); axsc.grid(alpha=0.3)
    fig.colorbar(sc, ax=axsc, label="speed [km/h]", pad=0.01)

    fig.tight_layout()
    out = INSPECT_DIR / "bottleneck_micro_snapshot.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved: {out.relative_to(_ROOT)}")


def main() -> None:
    if not NPZ.exists():
        sys.exit(f"Missing {NPZ}; run the 5-rollout 2000-vph inspection first.")
    macro()
    micro()


if __name__ == "__main__":
    main()
