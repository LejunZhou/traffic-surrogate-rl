"""
Post-hoc, multi-seed checkpoint selection for a PPO run (M7 §7.14 lesson:
the single-seed EvalCallback rewards luck on knife-edge demand cells).

1. Rank the run's checkpoints by the deterministic 18-cell eval that ran at
   the same step (eval/evaluations.npz): gridlock-free passes first, then mean.
2. Re-score the top-k candidates with scripts/eval_policy_grid_sumo.evaluate_policies
   (every cell x several seeds, ~12 min per checkpoint for 18 x 3).
3. Choose the best grid mean subject to breakdowns <= --max-breakdowns and copy
   it to <run_dir>/best_model_multiseed.zip; write a ranking JSON next to it.

Usage:
  python scripts/select_checkpoint_multiseed.py --run-dir runs/rl/<run> \\
      --config configs/rl/ppo_sumo_m7_run7_range.yaml --top-k 5 --seeds 0 1 2 \\
      --demands 1500 1600 1700 1800 1900 2000 --ramp-demands 400 600 800
  add --dry-run to only print the candidate ranking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

BREAKDOWN_EVAL = -120.0


def rank_candidates(run_dir: Path, top_k: int, extra: list[str]) -> list[dict]:
    d = np.load(run_dir / "eval" / "evaluations.npz")
    results, timesteps = d["results"], d["timesteps"]
    cands = []
    for t, row in zip(timesteps, results):
        ckpt = next(iter((run_dir / "checkpoints").glob(f"*_{int(t)}_steps.zip")), None)
        if ckpt is None:
            continue
        cands.append({"step": int(t), "path": str(ckpt), "det_mean": float(row.mean()),
                      "det_min": float(row.min()), "det_gridlocked": int((row < BREAKDOWN_EVAL).sum())})
    cands.sort(key=lambda c: (c["det_gridlocked"] > 0, -c["det_mean"]))
    chosen = cands[:top_k]
    best_step = int(timesteps[int(np.argmax(results.mean(1)))])   # what best_model.zip corresponds to
    for e in extra:
        p = run_dir / e
        if p.exists() and best_step not in {c["step"] for c in chosen}:
            chosen.append({"step": best_step, "path": str(p), "det_mean": float(results.mean(1).max()),
                           "det_min": float(results[int(np.argmax(results.mean(1)))].min()),
                           "det_gridlocked": int((results[int(np.argmax(results.mean(1)))] < BREAKDOWN_EVAL).sum())})
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed checkpoint selection")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--extra", nargs="*", default=["best_model.zip"], help="extra files in run dir to include")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--demands", type=float, nargs="+", default=[1500, 1600, 1700, 1800, 1900, 2000])
    ap.add_argument("--ramp-demands", type=float, nargs="+", default=[400, 600, 800])
    ap.add_argument("--max-breakdowns", type=int, default=3)
    ap.add_argument("--network-dir", default="data/raw/rl_network_ckpt_select")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cands = rank_candidates(args.run_dir, args.top_k, args.extra)
    print("== candidates (deterministic single-seed eval) ==")
    for c in cands:
        print(f"  step {str(c['step']):>6}  det mean {c['det_mean']:7.1f}  min {c['det_min']:7.1f}  gridlocked {c['det_gridlocked']:2d}  {Path(c['path']).name}")
    if args.dry_run:
        return

    from eval_policy_grid_sumo import evaluate_policies, print_summary  # noqa: E402
    out = args.run_dir / "eval" / "multiseed_selection.jsonl"
    summary = evaluate_policies([c["path"] for c in cands], args.demands, args.ramp_demands, args.seeds,
                                args.config, out, network_dir=args.network_dir, quiet=True)
    print_summary(summary)

    ranking = []
    for c in cands:
        s = summary[c["path"]]
        ranking.append({**c, "grid_mean": s["grid_mean"], "grid_min": s["grid_min"], "breakdowns": s["breakdowns"], "n": s["n"]})
    feasible = [r for r in ranking if r["breakdowns"] <= args.max_breakdowns]
    pool = feasible or ranking
    best = max(pool, key=lambda r: r["grid_mean"])
    ranking.sort(key=lambda r: (r["breakdowns"] > args.max_breakdowns, -r["grid_mean"]))
    print(f"\n== multi-seed ranking (feasible = breakdowns <= {args.max_breakdowns}/{best['n']}) ==")
    for r in ranking:
        flag = "*" if r is best else " "
        print(f" {flag} step {str(r['step']):>6}  grid mean {r['grid_mean']:7.1f}  min {r['grid_min']:7.1f}  breakdowns {r['breakdowns']:2d}/{r['n']}  det {r['det_mean']:7.1f}")
    dest = args.run_dir / "best_model_multiseed.zip"
    shutil.copy(best["path"], dest)
    (args.run_dir / "eval" / "multiseed_selection.json").write_text(json.dumps({"chosen": best, "ranking": ranking, "seeds": args.seeds,
                                                                              "max_breakdowns": args.max_breakdowns}, indent=1))
    print(f"\nchosen: step {best['step']} -> {dest}  (feasible: {bool(feasible)})")


if __name__ == "__main__":
    main()
