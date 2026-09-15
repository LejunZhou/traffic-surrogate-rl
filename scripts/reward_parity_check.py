"""
E2 — reward-parity check (draft §7.2): replay the E0 SUMO sweeps (constant u,
store-and-flush and feedforward schedules) through the ensemble-mean plant
model and compare each reward term's episode-sum range from SUMO with the
one from the surrogate. The surrogate is accepted for RL only if every term's
range agrees within 20 %.

  PYTHONPATH=src python scripts/reward_parity_check.py --ensemble runs/surrogate/plant_v2_round0 \\
      [--store data/plant_v2/round0] [--reward-config configs/rl/ppo_common.yaml] [--out _progress/m9_e2_parity.json]

Terms (unit weights): three-term S_out, S_que, S_std and the TTS pieces
S_road (veh h on the road), S_backlog (veh h of conservation backlog); the
ramp queue is identical in both (analytic in both envs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl.reward import RewardWeights, backlog_estimate  # noqa: E402
from sumo_env.rollout import load_rollout_npz, rescore_return  # noqa: E402
from sumo_env.rollout_store import RolloutStore  # noqa: E402
from surrogate.deeponet import DeepONetEnsemble  # noqa: E402
from utils.config import load_config  # noqa: E402

TERMS = ("S_out", "S_que", "S_std", "S_road", "S_backlog", "S_tts")


def unit_sums(density, outflow, queue, d, r, q_ref, dt=30.0, dx_km=0.1, skip=3) -> dict:
    dt_h = dt / 3600.0
    K = density.shape[1]
    std = density.std(axis=0)
    road = density.sum(axis=0) * dx_km
    off = np.cumsum((d + r) * dt_h); srv = np.cumsum(np.maximum(outflow, 0) * dt_h)
    backlog = np.maximum(off - srv - road - queue, 0.0)
    sl = slice(skip, K)
    return {"S_out": float(np.sum(np.maximum(0, q_ref[sl] - outflow[sl]) / q_ref[sl])),
            "S_que": float(np.sum((queue[sl] / 400.0) ** 2)), "S_std": float(np.sum(std[sl] / 6.0)),
            "S_road": float(np.sum(road[sl]) * dt_h), "S_backlog": float(np.sum(backlog[sl]) * dt_h),
            "S_tts": float(np.sum((road[sl] + queue[sl] + backlog[sl]) * dt_h))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--store", default="data/plant_v2/round0")
    ap.add_argument("--reward-config", default="configs/rl/ppo_common.yaml")
    ap.add_argument("--prefix", default="e0_", help="rollout file prefix of the sweeps")
    ap.add_argument("--exclude", default="e0_insertion", help="prefix to exclude")
    ap.add_argument("--out", default="_progress/m9_e2_parity.json")
    ap.add_argument("--tolerance", type=float, default=0.20)
    args = ap.parse_args()
    ens = DeepONetEnsemble.load(PROJECT_ROOT / args.ensemble)
    store = RolloutStore(PROJECT_ROOT / args.store)
    files = [e["file"] for e in store.entries if e["file"].startswith(args.prefix) and not e["file"].startswith(args.exclude)]
    weights = RewardWeights.from_config(load_config(str(PROJECT_ROOT / args.reward_config))["env"]["reward"])
    dx_km = float(ens.x_grid[1] - ens.x_grid[0]) / 1000.0
    rows = []
    for fn in files:
        arrays, meta = load_rollout_npz(store.root / fn)
        branch = ens.norm.branch_input(arrays["mainline_demand"], arrays["ramp_inflow_vph"])[None]
        rho_m, q_m = ens.predict_full(branch)
        rho_hat = np.clip(rho_m[:, 0].mean(0), 0, 143); q_hat = np.clip(q_m[:, 0].mean(0), 0, 3000)
        d, r, Q, qref = arrays["mainline_demand"], arrays["ramp_arrival"], arrays["ramp_queue"], arrays["q_ref"]
        s_sumo = unit_sums(arrays["density"], arrays["outflow_vph"], Q, d, r, qref, ens.dt, dx_km)
        s_sur = unit_sums(rho_hat, q_hat, Q, d, r, qref, ens.dt, dx_km)
        ret_sumo = rescore_return(arrays, weights, ens.dt, 90.0, dx_km)["return"]
        ret_sur = rescore_return(arrays, weights, ens.dt, 90.0, dx_km, density=rho_hat, outflow=q_hat)["return"]
        rows.append({"file": fn, "controller": meta["controller"]["type"], "sumo": s_sumo, "surrogate": s_sur,
                     "return_sumo": ret_sumo, "return_surrogate": ret_sur})
    report = {"n": len(rows), "terms": {}, "tolerance": args.tolerance, "reward_form": weights.form}
    print(f"\n== E2 reward parity: {len(rows)} E0 sweep rollouts through {args.ensemble} (ensemble mean) ==")
    print(f"  {'term':<10s} {'SUMO range':>12s} {'surr range':>12s} {'ratio':>7s} {'corr':>6s} {'mean abs diff':>14s}  ok")
    all_ok = True
    for t in TERMS:
        a = np.array([r["sumo"][t] for r in rows]); b = np.array([r["surrogate"][t] for r in rows])
        ra, rb = float(np.ptp(a)), float(np.ptp(b))
        ratio = rb / max(ra, 1e-9)
        ok = abs(ratio - 1.0) <= args.tolerance
        all_ok &= ok
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
        report["terms"][t] = {"sumo_range": ra, "surrogate_range": rb, "ratio": ratio, "corr": corr,
                              "mean_abs_diff": float(np.mean(np.abs(a - b))), "ok": ok}
        print(f"  {t:<10s} {ra:12.1f} {rb:12.1f} {ratio:7.3f} {corr:6.3f} {float(np.mean(np.abs(a - b))):14.2f}  {'ok' if ok else 'MISS'}")
    ret_s = np.array([r["return_sumo"] for r in rows]); ret_p = np.array([r["return_surrogate"] for r in rows])
    rel = np.abs(ret_p - ret_s) / np.maximum(np.abs(ret_s), 1e-6)
    report["return"] = {"mean_rel_err": float(rel.mean()), "median_rel_err": float(np.median(rel)),
                        "corr": float(np.corrcoef(ret_s, ret_p)[0, 1]), "range_ratio": float(np.ptp(ret_p) / max(np.ptp(ret_s), 1e-9))}
    # best constant / best schedule per profile agree?
    by_prof: dict[int, list] = {}
    for r in rows:
        parts = r["file"].split("_"); prof = int(parts[2]) if parts[1] != "insertion" else -1
        by_prof.setdefault(prof, []).append(r)
    agree = 0
    for prof, rs in by_prof.items():
        bs = max(rs, key=lambda r: r["return_sumo"])["file"]; bp = max(rs, key=lambda r: r["return_surrogate"])["file"]
        agree += bs == bp
    report["best_policy_agreement"] = agree / max(len(by_prof), 1)
    report["passed"] = bool(all_ok)
    print(f"  return ({weights.form}): mean rel err {rel.mean():.3f}, corr {report['return']['corr']:.3f}, range ratio "
          f"{report['return']['range_ratio']:.3f}; best-policy-per-profile agreement {agree}/{len(by_prof)}")
    print(f"  parity gate (every range within {args.tolerance:.0%}): {'PASSED' if all_ok else 'FAILED'}")
    out = PROJECT_ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "rows": rows}, indent=1))
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
