"""
Balance the three outflow-reward terms from a constant-u sweep (Milestone 7).

Reads the JSONL written by scripts/run_u_sweep_sumo.py (one line per
constant u, with per-step `outflow_vph_steps`, `queue_steps`, `std_steps`)
and, for a given set of normalisers (q_ref, queue_norm, sigma_ref):

1. recomputes each term's *unit-weight* episode sum for every u:
       S_out(u) = Σ_k max(0, q_ref - q_out_k) / q_ref
       S_que(u) = Σ_k (Q_k / queue_norm)^2
       S_std(u) = Σ_k std_k / sigma_ref
2. measures how much each sum varies across the sweep
   (range = max_u - min_u). A term whose range is much larger than the
   others' decides the optimum on its own — that is "one term dominating".
3. proposes weights that equalise the ranges,
       w_i = target_range / range_i,
   normalised so that beta == 1 (queue is the anchor), then reports the
   totals and per-term shares under the proposed weights, and which u wins.

The capacity-drop check is printed first: if q_out is monotone increasing
in u, metering cannot raise outflow at this demand and the interior
optimum (if any) comes from the std/queue trade, not from throughput.

Usage:
  python scripts/balance_reward_terms.py --sweep _progress/m7_u_sweep_seed0.jsonl
  python scripts/balance_reward_terms.py --sweep sweep.jsonl \\
      --q-ref auto --queue-norm 400 --sigma-ref 6 --target-range 100
  # Try explicit weights instead of the proposal:
  python scripts/balance_reward_terms.py --sweep sweep.jsonl --weights 3 1 1

No SUMO or torch needed; numpy only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TERMS = ("outflow", "queue", "std")


def load_sweep(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("u", r.get("action_mean", 0.0)))
    if not rows:
        raise ValueError(f"no sweep points in {path}")
    return rows


def unit_sums(row: dict, q_ref: float, queue_norm: float, sigma_ref: float, skip_steps: int) -> dict:
    q_out = np.asarray(row["outflow_vph_steps"], dtype=np.float64)[skip_steps:]
    queue = np.asarray(row["queue_steps"], dtype=np.float64)[skip_steps:]
    std = np.asarray(row["std_steps"], dtype=np.float64)[skip_steps:]
    return {
        "outflow": float(np.sum(np.maximum(0.0, q_ref - q_out) / q_ref)),
        "queue": float(np.sum((queue / queue_norm) ** 2)),
        "std": float(np.sum(std / sigma_ref)),
    }


def propose_weights(sums_by_u: list[dict], target_range: float | None) -> tuple[dict, dict]:
    ranges = {
        t: max(s[t] for s in sums_by_u) - min(s[t] for s in sums_by_u) for t in TERMS
    }
    if target_range is None:
        # geometric mean of the ranges: keeps the weights near 1 overall
        positive = [r for r in ranges.values() if r > 0]
        target_range = float(np.exp(np.mean(np.log(positive)))) if positive else 1.0
    raw = {t: (target_range / ranges[t]) if ranges[t] > 0 else 1.0 for t in TERMS}
    anchor = raw["queue"] if raw["queue"] > 0 else 1.0
    weights = {t: raw[t] / anchor for t in TERMS}
    return weights, ranges


def _episode_q_ref(row: dict, q_ref: float, mode: str) -> float:
    if mode == "offered":
        offered = float(row.get("demand_vph", float("nan"))) + float(row.get("ramp_demand_vph", float("nan")))
        if np.isfinite(offered):
            return float(min(offered, q_ref))
    return q_ref


def _grid_report(rows: list[dict], cells: list[tuple[float, float]], args) -> None:
    """Balance over a demand-grid sweep: ranges are taken across *all* episodes."""
    if str(args.q_ref).lower() == "auto":
        q_ref = float(max(np.mean(np.asarray(r["outflow_vph_steps"])[args.skip_steps:]) for r in rows))
    else:
        q_ref = float(args.q_ref)
    print(f"== Demand-grid sweep: {len(cells)} cells, {len(rows)} episodes; q_ref={q_ref:.0f} ({args.q_ref_mode}), "
          f"queue_norm={args.queue_norm:.0f}, sigma_ref={args.sigma_ref:.2f} ==")
    sums = [unit_sums(r, _episode_q_ref(r, q_ref, args.q_ref_mode), args.queue_norm, args.sigma_ref, args.skip_steps) for r in rows]
    weights, ranges = propose_weights(sums, args.target_range)
    print("\n== Range of each unit-weight episode sum across ALL grid episodes ==")
    for t in TERMS:
        print(f"  {t:8s} range={ranges[t]:9.1f}")
    print("\n== Proposed weights (equal ranges, anchored at beta=1) ==")
    print(f"  delta={weights['outflow']:.3f}  beta={weights['queue']:.3f}  gamma={weights['std']:.3f}")
    if args.weights is not None:
        weights = {"outflow": args.weights[0], "queue": args.weights[1], "std": args.weights[2]}
        print(f"\nEvaluating explicit weights: delta={weights['outflow']}, beta={weights['queue']}, gamma={weights['std']}")

    print("\n== Per cell: best constant u and per-term shares at that u (under these weights) ==")
    print(f"  {'mainline':>8} {'ramp':>5} {'best u':>6} {'return':>8} {'shares out/que/std':>20} {'max share':>9} {'worst-u':>8} {'return@worst':>12} {'max share anywhere':>18}")
    worst_share = 0.0
    for cell in cells:
        idx = [i for i, r in enumerate(rows) if (float(r.get("demand_vph", float("nan"))), float(r.get("ramp_demand_vph", float("nan")))) == cell]
        totals, shares = [], []
        for i in idx:
            terms = {t: weights[t] * sums[i][t] for t in TERMS}
            tot = sum(terms.values()); totals.append(-tot)
            shares.append({t: (terms[t] / tot if tot > 0 else 0.0) for t in TERMS})
        best = int(np.argmax(totals)); worst = int(np.argmin(totals))
        cell_max_share = max(max(sh.values()) for sh in shares)
        worst_share = max(worst_share, cell_max_share)
        u_best = float(rows[idx[best]].get("u", rows[idx[best]].get("action_mean")))
        u_worst = float(rows[idx[worst]].get("u", rows[idx[worst]].get("action_mean")))
        sh = shares[best]
        print(f"  {cell[0]:8.0f} {cell[1]:5.0f} {u_best:6.2f} {totals[best]:8.1f} "
              f"{sh['outflow']:6.2f}/{sh['queue']:4.2f}/{sh['std']:4.2f}      {max(sh.values()):9.2f} {u_worst:8.2f} {totals[worst]:12.1f} {cell_max_share:18.2f}")
    print(f"\n  max per-term share over the whole grid: {worst_share:.2f} (acceptance in the M7 plan: <= 0.6 at every constant u)")
    print("\nPaste into the env.reward block:")
    print(f"    delta: {weights['outflow']:.3f}\n    beta: {weights['queue']:.3f}\n    gamma: {weights['std']:.3f}\n"
          f"    q_ref: {q_ref:.1f}\n    queue_norm: {args.queue_norm:.1f}\n    sigma_ref: {args.sigma_ref:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance outflow/queue/std reward terms from a u-sweep")
    parser.add_argument("--sweep", type=Path, required=True, help="JSONL from run_u_sweep_sumo.py")
    parser.add_argument(
        "--q-ref",
        default="auto",
        help="Outflow reference (veh/h). 'auto' = max time-averaged outflow in the sweep",
    )
    parser.add_argument("--queue-norm", type=float, default=400.0)
    parser.add_argument("--sigma-ref", type=float, default=6.0)
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=3,
        help="Ignore the first N control steps (reward warmup; no vehicle has reached det_18 yet)",
    )
    parser.add_argument(
        "--target-range",
        type=float,
        default=None,
        help="Per-episode range each term should span across the sweep (default: geometric mean of measured ranges)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("DELTA", "BETA", "GAMMA"),
        default=None,
        help="Evaluate these explicit weights instead of the proposal",
    )
    parser.add_argument(
        "--q-ref-mode",
        choices=("fixed", "offered"),
        default="fixed",
        help="'fixed': one q_ref for all episodes (default). 'offered': per-episode q_ref = "
             "min(mainline + ramp arrivals, --q-ref) so the outflow term measures unserved demand "
             "instead of distance to a single capacity number (for demand-grid sweeps).",
    )
    args = parser.parse_args()

    rows = load_sweep(args.sweep)
    cells = sorted({(float(r.get("demand_vph", float("nan"))), float(r.get("ramp_demand_vph", float("nan")))) for r in rows})
    if len(cells) > 1:
        _grid_report(rows, cells, args)
        return
    us = [float(r.get("u", r.get("action_mean"))) for r in rows]

    # ---- capacity-drop check ----------------------------------------
    mean_out = [float(np.mean(np.asarray(r["outflow_vph_steps"])[args.skip_steps:])) for r in rows]
    peak_idx = int(np.argmax(mean_out))
    print("== Outflow vs u (time-averaged det_18 flow, veh/h) ==")
    for u, q in zip(us, mean_out):
        print(f"  u={u:.2f}  q_out={q:7.1f}")
    monotone = all(b >= a - 1e-9 for a, b in zip(mean_out, mean_out[1:]))
    if monotone or peak_idx == len(us) - 1:
        print(f"  -> q_out peaks at u={us[peak_idx]:.2f} (the top of the sweep): NO capacity drop at this demand.\n"
              "     Outflow alone cannot produce an interior optimum; only the std/queue trade can.")
    else:
        print(f"  -> q_out peaks at u*={us[peak_idx]:.2f} < 1: capacity drop present; outflow term is well-posed.")

    q_ref = float(max(mean_out)) if str(args.q_ref).lower() == "auto" else float(args.q_ref)
    print(f"\nNormalisers: q_ref={q_ref:.1f} veh/h, queue_norm={args.queue_norm:.0f} veh, "
          f"sigma_ref={args.sigma_ref:.2f} veh/km, skip_steps={args.skip_steps}")

    # ---- unit-weight sums -------------------------------------------
    sums_by_u = [unit_sums(r, q_ref, args.queue_norm, args.sigma_ref, args.skip_steps) for r in rows]
    print("\n== Unit-weight episode sums (delta=beta=gamma=1) ==")
    print(f"  {'u':>5} {'S_outflow':>10} {'S_queue':>10} {'S_std':>10}")
    for u, s in zip(us, sums_by_u):
        print(f"  {u:5.2f} {s['outflow']:10.1f} {s['queue']:10.1f} {s['std']:10.1f}")

    weights, ranges = propose_weights(sums_by_u, args.target_range)
    print("\n== Range across the sweep (max - min of each unit-weight sum) ==")
    for t in TERMS:
        print(f"  {t:8s} range={ranges[t]:8.1f}")
    print("\n== Proposed weights (equal ranges, anchored at beta=1) ==")
    print(f"  delta={weights['outflow']:.3f}  beta={weights['queue']:.3f}  gamma={weights['std']:.3f}")

    if args.weights is not None:
        weights = {"outflow": args.weights[0], "queue": args.weights[1], "std": args.weights[2]}
        print(f"\nEvaluating explicit weights: delta={weights['outflow']}, beta={weights['queue']}, gamma={weights['std']}")

    # ---- totals under the chosen weights ----------------------------
    print("\n== Episode return and per-term share under these weights ==")
    print(f"  {'u':>5} {'return':>9} {'outflow':>9} {'queue':>9} {'std':>9}   shares (out/que/std)   max share")
    best_u, best_ret = None, -np.inf
    for u, s in zip(us, sums_by_u):
        parts = {t: weights[t] * s[t] for t in TERMS}
        total = sum(parts.values())
        shares = {t: (parts[t] / total if total > 0 else 0.0) for t in TERMS}
        ret = -total
        if ret > best_ret:
            best_u, best_ret = u, ret
        print(f"  {u:5.2f} {ret:9.1f} {-parts['outflow']:9.1f} {-parts['queue']:9.1f} {-parts['std']:9.1f}   "
              f"{shares['outflow']:.2f}/{shares['queue']:.2f}/{shares['std']:.2f}   {max(shares.values()):.2f}")
    print(f"\n  best constant policy under these weights: u={best_u:.2f} (return {best_ret:.1f})")
    print("\nPaste into configs/rl/ppo_sumo.yaml env.reward:")
    print(f"    delta: {weights['outflow']:.3f}\n    beta: {weights['queue']:.3f}\n    gamma: {weights['std']:.3f}\n"
          f"    q_ref: {q_ref:.1f}\n    queue_norm: {args.queue_norm:.1f}\n    sigma_ref: {args.sigma_ref:.2f}")


if __name__ == "__main__":
    main()
