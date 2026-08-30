"""
Performance analysis of policies evaluated over a demand grid
(JSONL from scripts/eval_policy_grid_sumo.py / select_checkpoint_multiseed.py).

Produces, for each policy in the file(s):
  * per-cell table (mean +/- std over seeds, breakdowns, mean action, outflow, final queue)
  * action structure: mean u per (mainline, ramp) cell and linear sensitivities
    du/d(mainline 100 vph), du/d(ramp 100 vph)
  * within-episode behaviour: correlation of u with queue / density; u before vs after
    the first breakdown; recovery (density back < 40 veh/km) after a breakdown
  * throughput / queue trade vs the reference policies
  * figures: u-vs-cell heatmap, per-cell return bars vs references, example
    trajectories (u, queue, density) for chosen cells, breakdown timing
Usage:
  python scripts/analyze_policy_grid.py --eval _progress/m7_run7_grid_eval.jsonl \\
      --focus <policy-substring> --out-dir _progress/figures --tag m7_run7
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BREAKDOWN = 60.0
RECOVER = 40.0


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def cell_of(r):
    return (int(round(r["demand_vph"])), int(round(r["ramp_demand_vph"])))


def episode_stats(r):
    rho = np.asarray(r["mean_density_steps"]); u = np.asarray(r["action_steps"]); q = np.asarray(r["queue_steps"])
    # same criterion as the sweep summaries (max over detectors > 60 veh/km);
    # onset = first step whose corridor-mean density leaves free flow (> 40)
    out = {"breakdown": bool(r["density_max"] > BREAKDOWN)}
    if out["breakdown"]:
        k0 = int(np.argmax(rho > RECOVER)) if (rho > RECOVER).any() else int(np.argmax(rho)); out["t_break_min"] = k0 * 0.5
        out["u_before"] = float(u[max(0, k0 - 6):k0].mean()) if k0 > 0 else float("nan")
        out["u_after"] = float(u[k0:k0 + 10].mean())
        after = rho[k0:]; rec = np.where(after < RECOVER)[0]
        out["recovered"] = bool(rec.size) ; out["t_recover_min"] = float((k0 + rec[0]) * 0.5) if rec.size else float("nan")
    out["corr_u_queue"] = float(np.corrcoef(u[3:], q[3:])[0, 1]) if q[3:].std() > 1e-6 and u[3:].std() > 1e-6 else float("nan")
    out["corr_u_rho"] = float(np.corrcoef(u[3:], rho[3:])[0, 1]) if rho[3:].std() > 1e-6 and u[3:].std() > 1e-6 else float("nan")
    out["u_range"] = float(u.max() - u.min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--focus", default="learned", help="substring of the policy label to analyse in depth")
    ap.add_argument("--out-dir", type=Path, default=Path("_progress/figures"))
    ap.add_argument("--tag", default="analysis")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    rows = load(args.eval)
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)
    labels = {p: rs[0]["label"] for p, rs in by_policy.items()}
    cells = sorted({cell_of(r) for r in rows})
    mains = sorted({c[0] for c in cells}); ramps = sorted({c[1] for c in cells})

    # ---- per-cell tables ----
    table = {}
    for p, rs in by_policy.items():
        t = {}
        for c in cells:
            eps = [r for r in rs if cell_of(r) == c]
            if not eps:
                continue
            ret = np.array([e["total_reward"] for e in eps])
            t[c] = {"mean": ret.mean(), "std": ret.std(ddof=1) if len(ret) > 1 else 0.0, "n": len(eps),
                    "breakdowns": sum(e["density_max"] > BREAKDOWN for e in eps),
                    "u": np.mean([e["action_mean"] for e in eps]), "outflow": np.mean([e["outflow_vph_mean"] for e in eps]),
                    "queue_final": np.mean([e["queue_final"] for e in eps])}
        table[p] = t
    print("== per-cell mean return (breakdowns) ==")
    hdr = f"{'cell':>10}" + "".join(f"{labels[p][:20]:>24}" for p in by_policy)
    print(hdr)
    for c in cells:
        print(f"{c[0]}+{c[1]:<5}" + "".join(f"{table[p][c]['mean']:14.1f} ({table[p][c]['breakdowns']})  " for p in by_policy if c in table[p]))
    print(f"{'grid':>10}" + "".join(f"{np.mean([table[p][c]['mean'] for c in cells]):14.1f} ({sum(table[p][c]['breakdowns'] for c in cells)})  " for p in by_policy))

    focus = [p for p in by_policy if args.focus in labels[p] or args.focus in p]
    refs = [p for p in by_policy if p not in focus]
    report = {"cells": [list(c) for c in cells], "policies": {p: {"label": labels[p], "cells": {f"{c[0]}+{c[1]}": v for c, v in table[p].items()}} for p in by_policy}}

    for p in focus:
        rs = by_policy[p]
        print(f"\n== action structure: {labels[p]} ==")
        U = np.array([[table[p][(m, rp)]["u"] if (m, rp) in table[p] else np.nan for rp in ramps] for m in mains])
        corner = "mainline/ramp"
        print(f"{corner:>14}" + "".join(f"{rp:>8}" for rp in ramps))
        for m, row in zip(mains, U):
            print(f"{m:>14}" + "".join(f"{v:8.3f}" for v in row))
        X = np.array([[m, rp, 1.0] for m in mains for rp in ramps]); y = U.reshape(-1)
        ok = np.isfinite(y); coef = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        print(f"  linear fit: du = {coef[0]*100:+.4f} per +100 vph mainline, {coef[1]*100:+.4f} per +100 vph ramp arrivals")
        st = [episode_stats(r) for r in rs]
        cu = [s["corr_u_queue"] for s in st if np.isfinite(s["corr_u_queue"])]; cr = [s["corr_u_rho"] for s in st if np.isfinite(s["corr_u_rho"])]
        print(f"  within-episode: corr(u, queue) median {np.median(cu):+.2f}, corr(u, density) median {np.median(cr):+.2f}, "
              f"u range within episode median {np.median([s['u_range'] for s in st]):.3f}")
        bd = [s for s in st if s["breakdown"]]
        print(f"  breakdowns: {len(bd)}/{len(st)} episodes; onset median {np.median([s['t_break_min'] for s in bd]) if bd else float('nan'):.0f} min; "
              f"u before onset {np.mean([s['u_before'] for s in bd]) if bd else float('nan'):.3f} -> after {np.mean([s['u_after'] for s in bd]) if bd else float('nan'):.3f}; "
              f"recovered within the hour: {sum(s['recovered'] for s in bd)}/{len(bd)}")
        # throughput / queue trade vs references
        for q in refs:
            dq = np.mean([table[p][c]["queue_final"] - table[q][c]["queue_final"] for c in cells if c in table[q]])
            do = np.mean([table[p][c]["outflow"] - table[q][c]["outflow"] for c in cells if c in table[q]])
            dr = np.mean([table[p][c]["mean"] - table[q][c]["mean"] for c in cells if c in table[q]])
            wins = sum(table[p][c]["mean"] > table[q][c]["mean"] + 1.0 for c in cells if c in table[q])
            print(f"  vs {labels[q][:30]:30s}: return {dr:+6.1f} avg/cell, wins {wins}/{len(cells)} cells, outflow {do:+5.0f} vph, final queue {dq:+5.0f} veh")
        report["policies"][p]["action_matrix"] = U.tolist(); report["policies"][p]["u_sensitivity_per_100vph"] = [coef[0]*100, coef[1]*100]
        report["policies"][p]["breakdown_episodes"] = len(bd)

    if not args.no_figures:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # (a) per-cell return bars
        fig, ax = plt.subplots(figsize=(12, 4)); x = np.arange(len(cells)); w = 0.8 / len(by_policy)
        for i, p in enumerate(by_policy):
            ax.bar(x + i * w, [table[p][c]["mean"] for c in cells], w, yerr=[table[p][c]["std"] for c in cells], label=labels[p][:28], capsize=2)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([f"{c[0]}+{c[1]}" for c in cells], rotation=60, fontsize=8)
        ax.set_ylabel("episode return"); ax.legend(fontsize=7); ax.set_title(f"{args.tag}: per-cell return (mean ± std over seeds)")
        fig.tight_layout(); fig.savefig(args.out_dir / f"{args.tag}_cells.png", dpi=130); plt.close(fig)
        for p in focus:
            # (b) action heatmap
            U = np.array(report["policies"][p]["action_matrix"])
            fig, ax = plt.subplots(figsize=(4.5, 4)); im = ax.imshow(U, cmap="viridis", vmin=0.15, vmax=0.5)
            ax.set_xticks(range(len(ramps))); ax.set_xticklabels(ramps); ax.set_yticks(range(len(mains))); ax.set_yticklabels(mains)
            ax.set_xlabel("ramp arrivals (vph)"); ax.set_ylabel("mainline demand (vph)"); ax.set_title(f"mean u — {labels[p][:30]}", fontsize=9)
            for i in range(len(mains)):
                for j in range(len(ramps)):
                    ax.text(j, i, f"{U[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(args.out_dir / f"{args.tag}_u_heatmap.png", dpi=130); plt.close(fig)
            # (c) example trajectories: lowest-demand+800, mid, and the worst breakdown episode
            rs = by_policy[p]
            picks = [min((r for r in rs if cell_of(r) == (mains[0], ramps[-1])), key=lambda r: r["sumo_seed"]),
                     min((r for r in rs if cell_of(r) == (mains[len(mains)//2], ramps[1])), key=lambda r: r["sumo_seed"]),
                     min(rs, key=lambda r: r["total_reward"])]
            fig, axes = plt.subplots(3, len(picks), figsize=(4.2 * len(picks), 7.5), sharex=True)
            for j, r in enumerate(picks):
                t = np.arange(len(r["action_steps"])) * 0.5
                axes[0, j].plot(t, r["action_steps"]); axes[0, j].set_ylim(0, 1); axes[0, j].set_title(f"{cell_of(r)[0]}+{cell_of(r)[1]} seed {r['sumo_seed']} R={r['total_reward']:.0f}", fontsize=9)
                axes[1, j].plot(t, r["queue_steps"], color="tab:orange"); axes[2, j].plot(t, r["mean_density_steps"], color="tab:red"); axes[2, j].axhline(BREAKDOWN, ls="--", c="k", lw=0.8)
                axes[2, j].set_xlabel("min")
            axes[0, 0].set_ylabel("u"); axes[1, 0].set_ylabel("ramp queue (veh)"); axes[2, 0].set_ylabel("mean density (veh/km)")
            fig.suptitle(labels[p][:40], fontsize=9); fig.tight_layout(); fig.savefig(args.out_dir / f"{args.tag}_trajectories.png", dpi=130); plt.close(fig)
        print(f"\nfigures -> {args.out_dir}/{args.tag}_*.png")
    (args.out_dir / f"{args.tag}_analysis.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
