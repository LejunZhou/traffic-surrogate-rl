"""
Evaluate policies over a (mainline, ramp) demand grid with several SUMO seeds.

Companion to run_seed_sweep_sumo.py for the demand-range setting (M7 §7.12):
every policy is rolled out in every cell for every seed with the env config
of a PPO run (so speed_dev, observation layout, reward weights and the ramp
model match training). Writes one JSON line per episode and a summary with
per-cell means and the grid mean, plus breakdown counts (max mean density
> 60 veh/km).

Usage:
  python scripts/eval_policy_grid_sumo.py \\
      --policies runs/rl/<run>/best_model.zip u=0.25 u=0.3 \\
      --demands 1500 1600 1700 1800 1900 2000 --ramp-demands 400 600 800 \\
      --seeds 0 1 2 --config configs/rl/ppo_sumo_m7_run5_range.yaml \\
      --out _progress/m7_run5_grid_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from eval_sumo_baselines import _load_sumo_env_config, apply_sumo_overrides, rollout_policy_sumo  # noqa: E402
from rl.sumo_env_wrapper import SumoEnv  # noqa: E402

BREAKDOWN_DENSITY = 60.0


def summarize(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        p = out.setdefault(r["policy"], {"label": r["label"], "cells": {}})
        p["cells"].setdefault(f"{r['demand_vph']:.0f}+{r['ramp_demand_vph']:.0f}", []).append(r)
    summary = {}
    for policy, p in out.items():
        cells = {}
        for cell, eps in p["cells"].items():
            ret = np.array([e["total_reward"] for e in eps])
            cells[cell] = {
                "mean": float(ret.mean()), "std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
                "min": float(ret.min()), "n": len(eps),
                "breakdowns": int(sum(e["density_max"] > BREAKDOWN_DENSITY for e in eps)),
                "action_mean": float(np.mean([e["action_mean"] for e in eps])),
                "outflow_mean": float(np.mean([e["outflow_vph_mean"] for e in eps])),
            }
        all_eps = [e for eps in p["cells"].values() for e in eps]
        summary[policy] = {
            "label": p["label"], "cells": cells,
            "grid_mean": float(np.mean([e["total_reward"] for e in all_eps])),
            "grid_min": float(min(e["total_reward"] for e in all_eps)),
            "breakdowns": int(sum(e["density_max"] > BREAKDOWN_DENSITY for e in all_eps)),
            "n": len(all_eps),
        }
    return summary


def print_summary(summary: dict) -> None:
    policies = list(summary)
    cells = list(next(iter(summary.values()))["cells"])
    print(f"\n{'cell':>10} " + " ".join(f"{summary[p]['label'][:22]:>26}" for p in policies))
    for c in cells:
        print(f"{c:>10} " + " ".join(
            f"{summary[p]['cells'][c]['mean']:8.1f} ±{summary[p]['cells'][c]['std']:5.1f} u={summary[p]['cells'][c]['action_mean']:.2f} b{summary[p]['cells'][c]['breakdowns']}"
            for p in policies))
    print(f"{'grid mean':>10} " + " ".join(f"{summary[p]['grid_mean']:26.1f}" for p in policies))
    print(f"{'grid min':>10} " + " ".join(f"{summary[p]['grid_min']:26.1f}" for p in policies))
    print(f"{'breakdowns':>10} " + " ".join(f"{summary[p]['breakdowns']:>22d}/{summary[p]['n']}" for p in policies))


def evaluate_policies(policies, demands, ramp_demands, seeds, config, out, network_dir="data/raw/rl_network_grid_eval",
                      sumo_overrides=None, quiet=False) -> dict:
    """Roll every policy through every (mainline, ramp) cell for every seed; write JSONL + summary."""
    env_cfg = _load_sumo_env_config(overrides={}, config_path=Path(config))
    env_cfg["network_dir"] = network_dir
    env_cfg["demand_levels"] = [float(d) for d in demands]
    env_cfg["ramp_demand_levels"] = [float(r) for r in ramp_demands]
    apply_sumo_overrides(env_cfg, sumo_overrides)
    env = SumoEnv(env_cfg)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        with out.open("w", encoding="utf-8") as f:
            for policy in policies:
                for d in demands:
                    for rp in ramp_demands:
                        for seed in seeds:
                            r = rollout_policy_sumo(
                                policy_arg=str(policy), seed=int(seed), n_episodes=1, config_path=Path(config), env=env,
                                reset_options={"demand_vph": float(d), "ramp_demand_vph": float(rp), "sumo_seed": int(seed)},
                            )
                            r["sumo_seed"] = int(seed)
                            f.write(json.dumps(r) + "\n"); f.flush(); rows.append(r)
                            if not quiet:
                                print(f"  {r['label'][:24]:24s} {d:.0f}+{rp:.0f} seed={seed} return={r['total_reward']:8.1f} "
                                      f"u={r['action_mean']:.2f} rho_max={r['density_max']:4.0f}", flush=True)
    finally:
        env.close()
    summary = summarize(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy evaluation over a demand grid with several SUMO seeds")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--demands", type=float, nargs="+", required=True)
    ap.add_argument("--ramp-demands", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sumo-override", action="append", default=None, metavar="SECTION.KEY=VALUE")
    ap.add_argument("--network-dir", default="data/raw/rl_network_grid_eval")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = evaluate_policies(args.policies, args.demands, args.ramp_demands, args.seeds, args.config, args.out,
                                network_dir=args.network_dir, sumo_overrides=args.sumo_override)
    print_summary(summary)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
