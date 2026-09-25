"""
Field-level figures for a plant-model ensemble on a held-out split (M9).

  PYTHONPATH=src python scripts/plot_plant_eval.py \\
      --ensemble runs/surrogate/plant_v2_round0 --split test \\
      --out _progress/figures/m9_plant_eval

Writes
  fig_a_fields.png        SUMO density / ensemble mean / |error| / ensemble std for four
                          representative rollouts (median return error inside each regime)
  fig_b_exit_flow.png     inputs (demand, ramp inflow) and exit flow, truth vs ensemble +-2 std
  fig_c_cells.png         density at 1000 / 1300 / 1700 m vs time for the same rollouts
  fig_d_error_map.png     mean |error| and mean ensemble std over the whole split, (x, t)
  fig_e_return_scatter.png predicted vs true episode return per regime + calibration bins
  fig_f_case_studies.png  per rollout: demand, control input, SUMO vs predicted density, exit flow
  summary.json            the per-regime numbers used in the text

The representative rollouts are picked by the median return-prediction error inside
each (regime, breakdown) cell, never by the best case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sumo_env.rollout import breakdown_flags, load_rollout_npz  # noqa: E402
from surrogate.deeponet import DeepONetEnsemble  # noqa: E402
from surrogate.eval_plant import calibration  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REGIME_SLOT = {"store_flush": 0, "constant": 1, "alinea": 2, "random_signal": 3, "legacy_policy": 4, "feedforward": 5,
               "alinea_wide": 2, "alinea_dither": 6, "random_policy": 7}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MERGE_M = 1300.0

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def _pick_representatives(rows: list[dict], cells: list[tuple[str, bool]], min_n: int = 4) -> list[dict]:
    picked = []
    for group, bd in cells:
        cand = [r for r in rows if r["group"] == group and bool(r["breakdown_true"]) == bd]
        if not cand:
            continue
        cand.sort(key=lambda r: r["return_rel_err"])
        picked.append(cand[len(cand) // 2])
    if len(picked) < min_n and len(rows) >= min_n:
        # a split with one regime (e.g. the on-policy rollouts of an aggregation round): show the
        # return-error quantiles instead of one median case -- median, 75 %, 90 % and the worst
        cand = sorted(rows, key=lambda r: r["return_rel_err"])
        qs = [0.5, 0.75, 0.9, 1.0][: min_n]
        picked = []
        for q in qs:
            r = dict(cand[min(int(round(q * (len(cand) - 1))), len(cand) - 1)])
            r["group"] = f"{r['group']} q{int(q * 100)}"
            if r["file"] not in [x["file"] for x in picked]:
                picked.append(r)
    return picked


def _predict(ens: DeepONetEnsemble, store: Path, fn: str):
    arrays, meta = load_rollout_npz(store / fn)
    branch = ens.norm.branch_input(arrays["mainline_demand"], arrays["ramp_inflow_vph"])[None]
    rho_m, q_m = ens.predict_full(branch)
    rho_m, q_m = np.maximum(rho_m[:, 0], 0.0), np.maximum(q_m[:, 0], 0.0)
    return arrays, meta, rho_m, q_m


def fig_fields(ens, store, reps, out: Path) -> None:
    n = len(reps)
    fig, axes = plt.subplots(n, 4, figsize=(13, 2.6 * n), dpi=150, constrained_layout=True, squeeze=False)
    t_min = np.arange(ens.K) * ens.dt / 60.0
    extent = [t_min[0], t_min[-1] + ens.dt / 60.0, ens.x_grid[0] - 50, ens.x_grid[-1] + 50]
    for i, r in enumerate(reps):
        arrays, meta, rho_m, q_m = _predict(ens, store, r["file"])
        true = arrays["density"].astype(np.float64)
        pred, std = rho_m.mean(0), rho_m.std(0)
        vmax = max(float(true.max()), float(pred.max()), 60.0)
        panels = [(true, "SUMO density", "Blues", 0, vmax), (pred, "ensemble mean", "Blues", 0, vmax),
                  (np.abs(pred - true), "|error|", "Oranges", 0, max(20.0, np.abs(pred - true).max() * 0.8)),
                  (std, "ensemble std", "Purples", 0, max(5.0, std.max()))]
        for j, (field, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=extent,
                           interpolation="nearest")
            ax.axhline(MERGE_M, color=TEXT, lw=0.8, ls="--", alpha=0.6)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                bd = "breakdown" if r["breakdown_true"] else "no breakdown"
                ax.set_ylabel(f"{r['group']} · {bd}\nx (m)", fontsize=8)
                ax.text(0.02, 0.96, f"return true {r['return_true']:.1f}  pred {r['return_pred']:.1f}",
                        transform=ax.transAxes, fontsize=7.5, va="top", color=TEXT,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            if i == n - 1:
                ax.set_xlabel("time (min)")
    fig.suptitle("Plant DeepONet ensemble on held-out test rollouts: density field (veh/km/lane); dashed line = on-ramp merge at 1300 m",
                 fontsize=10)
    fig.savefig(out / "fig_a_fields.png"); fig.savefig(out / "fig_a_fields.pdf"); plt.close(fig)


def fig_exit_flow(ens, store, reps, out: Path) -> None:
    n = len(reps)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 5.2), dpi=150, constrained_layout=True, sharex=True, squeeze=False)
    t_min = np.arange(ens.K) * ens.dt / 60.0
    for i, r in enumerate(reps):
        arrays, meta, rho_m, q_m = _predict(ens, store, r["file"])
        ax = axes[0, i]
        ax.plot(t_min, arrays["mainline_demand"], color=PALETTE[0], lw=2, label="mainline demand")
        ax.plot(t_min, arrays["ramp_inflow_vph"], color=PALETTE[1], lw=2, label="ramp inflow (metered)")
        ax.plot(t_min, arrays["ramp_arrival"], color=PALETTE[1], lw=1, ls=":", label="ramp arrivals")
        ax.set_ylim(0, 3000); ax.grid(True)
        ax.set_title(f"{r['group']} · {'breakdown' if r['breakdown_true'] else 'no breakdown'}", fontsize=9)
        if i == 0:
            ax.set_ylabel("branch inputs (veh/h)"); ax.legend(fontsize=7, loc="upper right")
        ax = axes[1, i]
        q_true, q_pred, q_std = arrays["outflow_vph"].astype(np.float64), q_m.mean(0), q_m.std(0)
        ax.fill_between(t_min, q_pred - 2 * q_std, q_pred + 2 * q_std, color=PALETTE[2], alpha=0.2, lw=0,
                        label="ensemble ±2 std")
        ax.plot(t_min, q_true, color=TEXT, lw=2, label="SUMO exit flow")
        ax.plot(t_min, q_pred, color=PALETTE[2], lw=2, label="ensemble mean")
        rel = np.linalg.norm(q_pred - q_true) / max(np.linalg.norm(q_true), 1e-6)
        ax.text(0.02, 0.05, f"rel-L2 {rel:.3f}", transform=ax.transAxes, fontsize=8)
        ax.set_ylim(0, 2600); ax.grid(True); ax.set_xlabel("time (min)")
        if i == 0:
            ax.set_ylabel("exit flow (veh/h)"); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Exit flow prediction on the same held-out rollouts (top: the two branch inputs)", fontsize=10)
    fig.savefig(out / "fig_b_exit_flow.png"); fig.savefig(out / "fig_b_exit_flow.pdf"); plt.close(fig)


def fig_cells(ens, store, reps, out: Path, cells_m=(1000.0, 1300.0, 1700.0)) -> None:
    n = len(reps)
    fig, axes = plt.subplots(n, len(cells_m), figsize=(3.6 * len(cells_m), 2.3 * n), dpi=150,
                             constrained_layout=True, sharex=True, squeeze=False)
    t_min = np.arange(ens.K) * ens.dt / 60.0
    idx = [int(np.argmin(np.abs(ens.x_grid - c))) for c in cells_m]
    for i, r in enumerate(reps):
        arrays, meta, rho_m, q_m = _predict(ens, store, r["file"])
        true = arrays["density"].astype(np.float64)
        pred, std = rho_m.mean(0), rho_m.std(0)
        for j, ci in enumerate(idx):
            ax = axes[i, j]
            ax.fill_between(t_min, pred[ci] - 2 * std[ci], pred[ci] + 2 * std[ci], color=PALETTE[0], alpha=0.18, lw=0)
            for m in range(rho_m.shape[0]):
                ax.plot(t_min, rho_m[m, ci], color=PALETTE[0], lw=0.6, alpha=0.5)
            ax.plot(t_min, true[ci], color=TEXT, lw=2, label="SUMO")
            ax.plot(t_min, pred[ci], color=PALETTE[0], lw=2, label="ensemble mean")
            ax.axhline(60, color=PALETTE[7], lw=0.8, ls="--", alpha=0.7)
            ax.set_ylim(0, 150); ax.grid(True)
            if i == 0:
                ax.set_title(f"x = {ens.x_grid[ci]:.0f} m" + (" (merge)" if abs(ens.x_grid[ci] - MERGE_M) < 1 else ""),
                             fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{r['group']}\nρ (veh/km/lane)", fontsize=8)
            if i == n - 1:
                ax.set_xlabel("time (min)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Density at three detectors: SUMO (black), ensemble mean (blue), thin lines = the five members, band = ±2 std; "
                 "dashed red = 60 veh/km breakdown threshold", fontsize=9)
    fig.savefig(out / "fig_c_cells.png"); fig.savefig(out / "fig_c_cells.pdf"); plt.close(fig)


def fig_error_map(ens, store, files, out: Path) -> dict:
    err = np.zeros((ens.Nx, ens.K)); std = np.zeros_like(err); true_mean = np.zeros_like(err)
    err_q = np.zeros(ens.K); q_true_mean = np.zeros(ens.K)
    for fn in files:
        arrays, meta, rho_m, q_m = _predict(ens, store, fn)
        true = arrays["density"].astype(np.float64)
        err += np.abs(rho_m.mean(0) - true); std += rho_m.std(0); true_mean += true
        err_q += np.abs(q_m.mean(0) - arrays["outflow_vph"]); q_true_mean += arrays["outflow_vph"]
    n = len(files)
    err /= n; std /= n; true_mean /= n; err_q /= n; q_true_mean /= n
    t_min = np.arange(ens.K) * ens.dt / 60.0
    extent = [t_min[0], t_min[-1] + ens.dt / 60.0, ens.x_grid[0] - 50, ens.x_grid[-1] + 50]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2), dpi=150, constrained_layout=True)
    for ax, field, title, cmap in [(axes[0], true_mean, "mean SUMO density", "Blues"),
                                   (axes[1], err, "mean |error|", "Oranges"), (axes[2], std, "mean ensemble std", "Purples")]:
        im = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap, extent=extent, interpolation="nearest")
        ax.axhline(MERGE_M, color=TEXT, lw=0.8, ls="--", alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        ax.set_title(f"{title} (veh/km/lane)", fontsize=9); ax.set_xlabel("time (min)")
    axes[0].set_ylabel("x (m)")
    ax = axes[3]
    ax.plot(t_min, err.mean(0), color=PALETTE[1], lw=2, label="all cells")
    ax.plot(t_min, err[np.argmin(np.abs(ens.x_grid - MERGE_M))], color=PALETTE[7], lw=1.5, label="merge cell (1300 m)")
    ax.set_ylabel("|error| (veh/km/lane)"); ax.set_xlabel("time (min)"); ax.grid(True); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"density error vs time, {n} rollouts", fontsize=9)
    ax = axes[4]
    ax.plot(t_min, err_q, color=PALETTE[2], lw=2)
    ax.set_ylabel("|error| (veh/h)"); ax.set_xlabel("time (min)"); ax.grid(True); ax.set_ylim(0, None)
    ax.set_title(f"exit-flow error vs time, {n} rollouts", fontsize=9)
    ax.text(0.98, 0.95, f"mean SUMO exit flow {q_true_mean.mean():.0f} veh/h", transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color=MUTED)
    fig.suptitle("Where the plant model errs: averages over the whole held-out split", fontsize=10)
    fig.savefig(out / "fig_d_error_map.png"); fig.savefig(out / "fig_d_error_map.pdf"); plt.close(fig)
    return {"mean_abs_err_density": float(err.mean()), "mean_abs_err_at_merge": float(err[np.argmin(np.abs(ens.x_grid - MERGE_M))].mean()),
            "mean_abs_err_flow_vph": float(err_q.mean()), "mean_true_density": float(true_mean.mean()),
            "mean_true_flow_vph": float(q_true_mean.mean()), "per_cell_abs_err": err.mean(1).tolist(),
            "x_grid_m": ens.x_grid.tolist()}


def _draw_case_panels(fig, ens, store, r: dict, ax_demand, ax_u, ax_ramp, ax_true, ax_pred, ax_flow, first_col: bool = True,
                      title: bool = True) -> None:
    """Draw the six case-study panels of one rollout into the given axes."""
    arrays, meta, rho_m, q_m = _predict(ens, store, r["file"])
    t_min = np.arange(ens.K) * ens.dt / 60.0
    extent = [t_min[0], t_min[-1] + ens.dt / 60.0, ens.x_grid[0] - 50, ens.x_grid[-1] + 50]
    true = arrays["density"].astype(np.float64); pred = rho_m.mean(0)
    vmax = max(float(true.max()), float(pred.max()), 60.0)
    ax = ax_demand
    ax.plot(t_min, arrays["mainline_demand"], color=PALETTE[0], lw=2, label="mainline demand d(t)")
    ax.plot(t_min, arrays["ramp_arrival"], color=PALETTE[1], lw=2, label="ramp arrivals r(t)")
    ax.plot(t_min, arrays["mainline_demand"] + arrays["ramp_arrival"], color=MUTED, lw=1, ls="--", label="total offered")
    ax.axhline(2500, color=PALETTE[7], lw=0.8, ls=":", label="≈ merge capacity")
    ax.set_ylim(0, 3200); ax.grid(True)
    if title:
        ax.set_title(f"{r['group']} · {'breakdown' if r['breakdown_true'] else 'no breakdown'}\n{Path(r['file']).name}", fontsize=8.5)
    if first_col: ax.set_ylabel("demand (veh/h)"); ax.legend(fontsize=6.5, loc="lower right")
    ax = ax_u
    ax.step(t_min, arrays["action"], where="post", color=TEXT, lw=1.6)
    ax.set_ylim(-0.02, 1.02); ax.grid(True)
    if first_col: ax.set_ylabel("metering rate u")
    ax = ax_ramp
    ax.plot(t_min, arrays["ramp_arrival"], color=PALETTE[1], lw=1, ls=":", label="ramp arrivals")
    ax.plot(t_min, arrays["ramp_inflow_vph"], color=PALETTE[1], lw=1.6, label="ramp inflow (metered)")
    ax.set_ylim(0, 1400); ax.grid(True)
    ax.text(0.98, 0.93, f"max ramp queue {arrays['ramp_queue'].max():.0f} veh", transform=ax.transAxes, fontsize=7, ha="right", va="top", color=MUTED)
    if first_col: ax.set_ylabel("ramp flow (veh/h)"); ax.legend(fontsize=6.5, loc="upper left")
    for ax, field, label, ylab in [(ax_true, true, "SUMO", "SUMO density"),
                                   (ax_pred, pred, f"DeepONet  (rel-L2 {r['rel_l2_density']:.3f})", "DeepONet ensemble mean")]:
        im = ax.imshow(field, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.axhline(MERGE_M, color=TEXT, lw=0.8, ls="--", alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=8, va="top", bbox=dict(fc="white", ec="none", alpha=0.8))
        if first_col: ax.set_ylabel(ylab + "\nx (m)", fontsize=8.5)
    ax = ax_flow
    q_true, q_pred, q_std = arrays["outflow_vph"].astype(np.float64), q_m.mean(0), q_m.std(0)
    ax.fill_between(t_min, q_pred - 2 * q_std, q_pred + 2 * q_std, color=PALETTE[2], alpha=0.2, lw=0, label="ensemble ±2 std")
    ax.plot(t_min, q_true, color=TEXT, lw=1.8, label="SUMO exit flow")
    ax.plot(t_min, q_pred, color=PALETTE[2], lw=1.8, label="DeepONet")
    ax.set_ylim(0, 2700); ax.grid(True); ax.set_xlabel("time (min)")
    ax.text(0.02, 0.06, f"rel-L2 {r['rel_l2_flow']:.3f}\nreturn true {r['return_true']:.1f} / pred {r['return_pred']:.1f}", transform=ax.transAxes, fontsize=7.5)
    if first_col: ax.set_ylabel("exit flow (veh/h)"); ax.legend(fontsize=6.5, loc="center right")


def fig_case_studies(ens, store, reps, out: Path, stem: str = "fig_f_case_studies") -> None:
    """One column per rollout: demand, metering rate, ramp inflow, SUMO density, predicted density, exit flow."""
    n = len(reps)
    fig, axes = plt.subplots(6, n, figsize=(3.6 * n, 14.5), dpi=150, constrained_layout=True, sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [1.1, 0.7, 0.9, 1.4, 1.4, 1.1]})
    for i, r in enumerate(reps):
        _draw_case_panels(fig, ens, store, r, *axes[:, i], first_col=(i == 0))
    fig.suptitle("Held-out test rollouts: demand → metering rate → ramp inflow → SUMO density vs DeepONet prediction → exit flow (dashed line = merge at 1300 m)", fontsize=10)
    fig.savefig(out / f"{stem}.png"); fig.savefig(out / f"{stem}.pdf"); plt.close(fig)


def fig_case_single(ens, store, r: dict, out: Path, stem: str) -> None:
    """Landscape single-rollout case study, one row of four panels sharing the time axis:
    inputs (mainline demand, ramp arrivals, metered ramp inflow) | SUMO density | DeepONet density | exit flow."""
    FS_LAB, FS_TICK, FS_LEG = 12, 10.5, 10.5
    arrays, meta, rho_m, q_m = _predict(ens, store, r["file"])
    t_min = np.arange(ens.K) * ens.dt / 60.0
    extent = [t_min[0], t_min[-1] + ens.dt / 60.0, ens.x_grid[0] - 50, ens.x_grid[-1] + 50]
    true = arrays["density"].astype(np.float64); pred = rho_m.mean(0)
    vmax = max(float(true.max()), float(pred.max()), 60.0)
    fig, (ax_in, ax_true, ax_pred, ax_flow) = plt.subplots(1, 4, figsize=(17.0, 4.4), dpi=150, constrained_layout=True,
                                                            sharex=True, gridspec_kw={"width_ratios": [1.0, 1.15, 1.15, 1.0]})
    ax = ax_in
    ax.plot(t_min, arrays["mainline_demand"], color=PALETTE[0], lw=2, label="mainline demand")
    ax.plot(t_min, arrays["ramp_arrival"], color=PALETTE[3], lw=2, label="ramp arrivals")
    ax.plot(t_min, arrays["ramp_inflow_vph"], color=PALETTE[1], lw=1.8, label="ramp inflow (metered)")
    ax.set_ylim(0, 2700); ax.grid(True); ax.set_xlabel("time (min)", fontsize=FS_LAB); ax.set_ylabel("flow (veh/h)", fontsize=FS_LAB)
    ax.legend(fontsize=FS_LEG, loc="upper left", ncol=1)
    for ax, field, label in [(ax_true, true, "SUMO"), (ax_pred, pred, "DeepONet")]:
        im = ax.imshow(field, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.axhline(MERGE_M, color=TEXT, lw=0.8, ls="--", alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=FS_TICK)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=FS_LAB, va="top", bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_xlabel("time (min)", fontsize=FS_LAB); ax.set_ylabel("x (m)", fontsize=FS_LAB)
    ax = ax_flow
    q_true, q_pred, q_std = arrays["outflow_vph"].astype(np.float64), q_m.mean(0), q_m.std(0)
    ax.plot(t_min, q_true, color=TEXT, lw=1.8, label="SUMO exit flow")
    ax.plot(t_min, q_pred, color=PALETTE[2], lw=1.8, label="DeepONet")
    ax.set_ylim(0, 2700); ax.grid(True); ax.set_xlabel("time (min)", fontsize=FS_LAB); ax.set_ylabel("exit flow (veh/h)", fontsize=FS_LAB)
    ax.legend(fontsize=FS_LEG, loc="center right")
    fig.suptitle(f"{r['group']} · {'breakdown' if r['breakdown_true'] else 'no breakdown'} · "
                 f"{Path(r['file']).name}   (dashed line = merge at 1300 m)", fontsize=FS_LAB + 1)
    for ax in (ax_in, ax_true, ax_pred, ax_flow): ax.tick_params(labelsize=FS_TICK)
    fig.savefig(out / f"{stem}.png"); fig.savefig(out / f"{stem}.pdf"); plt.close(fig)


def fig_return_scatter(rows: list[dict], summary: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150, constrained_layout=True)
    ax = axes[0]
    groups = sorted({r["group"] for r in rows}, key=lambda g: REGIME_SLOT.get(g, 9))
    lo = min(min(r["return_true"] for r in rows), min(r["return_pred"] for r in rows)) * 1.05
    ax.plot([lo, 0], [lo, 0], color=MUTED, lw=1)
    ax.fill_between([lo, 0], [lo * 1.1, 0], [lo * 0.9, 0], color=GRID, alpha=0.8, lw=0, label="±10 %")
    for g in groups:
        rr = [r for r in rows if r["group"] == g]
        c = PALETTE[REGIME_SLOT.get(g, 6)]
        ax.scatter([r["return_true"] for r in rr], [r["return_pred"] for r in rr], s=22, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=f"{g} (n={len(rr)})",
                   marker="o" if not any(r["breakdown_true"] for r in rr) else "o")
    bd = [r for r in rows if r["breakdown_true"]]
    ax.scatter([r["return_true"] for r in bd], [r["return_pred"] for r in bd], s=60, facecolor="none",
               edgecolor=TEXT, linewidth=0.7, label="SUMO breakdown")
    ax.set_xlabel("true episode return (SUMO fields), veh h"); ax.set_ylabel("predicted return (ensemble-mean fields)")
    ax.set_title(f"episode return: mean rel. error {summary['all']['return_rel_err_mean']:.3f}, median {summary['all']['return_rel_err_median']:.3f}",
                 fontsize=9)
    ax.grid(True); ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for g in groups:
        rr = [r for r in rows if r["group"] == g]
        c = PALETTE[REGIME_SLOT.get(g, 6)]
        ax.scatter([r["return_spread"] for r in rr], [r["return_err"] for r in rr], s=22, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=g)
    m = max(max(r["return_spread"] for r in rows), max(r["return_err"] for r in rows)) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=1, label="error = spread")
    ax.set_xlabel("member spread of the return (std over 5 members), veh h"); ax.set_ylabel("|error| of the ensemble-mean return, veh h")
    ax.set_title("does the ensemble know when it is wrong? (per rollout)", fontsize=9); ax.grid(True); ax.legend(fontsize=7)
    ax = axes[2]
    bins = summary["calibration"]["bins"]
    ax.plot([b["std_mean"] for b in bins], [b["abs_err_mean"] for b in bins], color=PALETTE[4], lw=2, marker="o", ms=5,
            label="binned mean |error|")
    ax.plot([b["std_mean"] for b in bins], [b["abs_err_rms"] for b in bins], color=PALETTE[4], lw=1.2, ls="--", label="binned RMS error")
    sm = max(b["std_mean"] for b in bins)
    ax.plot([0, sm], [0, sm], color=MUTED, lw=1, label="1:1")
    ax.plot([0, sm], [0, summary["calibration"]["slope"] * sm], color=TEXT, lw=1, ls=":",
            label=f"fitted slope {summary['calibration']['slope']:.2f}")
    ax.set_xlabel("ensemble std of density (veh/km/lane), decile bins"); ax.set_ylabel("|error| of ensemble mean (veh/km/lane)")
    ax.set_title(f"density calibration: coverage of ±2 std = {summary['calibration']['coverage_2std']:.2f}", fontsize=9)
    ax.grid(True); ax.legend(fontsize=7)
    fig.savefig(out / "fig_e_return_scatter.png"); fig.savefig(out / "fig_e_return_scatter.pdf"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", default=None, help="eval_<split>_rows.jsonl from eval_surrogate_regimes (default: inside the ensemble dir)")
    ap.add_argument("--out", default="_progress/figures/m9_plant_eval")
    ap.add_argument("--case-pick", default="median", choices=["median", "busiest", "worst"],
                    help="with --case: median return error (default), highest peak offered demand, or largest return error")
    ap.add_argument("--case", default=None, metavar="GROUP[:breakdown|nobreakdown]",
                    help="write only a one-column case-study figure (fig_f_case_<group>_<bd>.png) for the median-error "
                         "rollout of that (regime, breakdown) cell, e.g. alinea_wide:breakdown; skips all other figures")
    args = ap.parse_args()
    ens = DeepONetEnsemble.load(PROJECT_ROOT / args.ensemble)
    store = Path(ens.manifest.get("store_dir", "data/plant_v2/round0"))
    store = store if store.is_absolute() else PROJECT_ROOT / store
    ens_dir = PROJECT_ROOT / args.ensemble
    rows_path = Path(args.rows) if args.rows else ens_dir / f"eval_{args.split}_rows.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    summary = json.loads((ens_dir / f"eval_{args.split}.json").read_text())
    out = PROJECT_ROOT / args.out; out.mkdir(parents=True, exist_ok=True)

    if args.case:
        group, _, bd_s = args.case.partition(":")
        bd = (bd_s or "breakdown").lower() != "nobreakdown"
        cand = [r for r in rows if r["group"] == group and bool(r["breakdown_true"]) == bd]
        if not cand:
            sys.exit(f"no rollout with group={group!r} and breakdown_true={bd} in {rows_path}")
        if args.case_pick == "median":
            reps1 = _pick_representatives(rows, [(group, bd)], min_n=1)
        elif args.case_pick == "busiest":
            reps1 = [max(cand, key=lambda r: r["peak_total"])]
        else:
            reps1 = [max(cand, key=lambda r: r["return_rel_err"])]
        stem = f"fig_f_case_{group}_{'breakdown' if bd else 'nobreakdown'}" + ("" if args.case_pick == "median" else f"_{args.case_pick}")
        print("case rollout:", reps1[0]["file"], round(reps1[0]["return_rel_err"], 3))
        fig_case_single(ens, store, reps1[0], out, stem=stem)
        print(f"written {out / stem}.png"); return

    groups_present = sorted({r["group"] for r in rows}, key=lambda g: REGIME_SLOT.get(g, 9))
    def cells(preferred):
        out = []
        for g, bd in preferred:
            if g in groups_present and any(r["group"] == g and bool(r["breakdown_true"]) == bd for r in rows):
                out.append((g, bd))
        for g in groups_present:                       # regimes not in the preferred list (e.g. the M13 controllers)
            if all(c[0] != g for c in out):
                bd = any(r["group"] == g and r["breakdown_true"] for r in rows)
                out.append((g, bd))
        return out
    reps = _pick_representatives(rows, cells([("store_flush", True), ("alinea", True), ("random_signal", True), ("legacy_policy", False)])[:4])
    reps6 = _pick_representatives(rows, cells([("constant", True), ("store_flush", True), ("feedforward", True), ("alinea", True),
                                               ("random_signal", True), ("legacy_policy", False)])[:7])
    print("representative rollouts:", [(r["file"], r["group"], r["breakdown_true"], round(r["return_rel_err"], 3)) for r in reps])
    fig_fields(ens, store, reps, out)
    fig_exit_flow(ens, store, reps, out)
    fig_cells(ens, store, reps, out)
    maps = fig_error_map(ens, store, [r["file"] for r in rows], out)
    fig_return_scatter(rows, summary, out)
    fig_case_studies(ens, store, reps6, out)
    (out / "summary.json").write_text(json.dumps({
        "ensemble": args.ensemble, "split": args.split, "n": len(rows),
        "representatives": [{k: r[k] for k in ("file", "group", "breakdown_true", "breakdown_pred", "return_true", "return_pred",
                                                "rel_l2_density", "rel_l2_flow", "return_rel_err")} for r in reps],
        "all": summary["all"], "by_group": summary["by_group"], "gate": summary["gate"],
        "calibration": {k: v for k, v in summary["calibration"].items() if k != "bins"}, "maps": maps}, indent=1))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
