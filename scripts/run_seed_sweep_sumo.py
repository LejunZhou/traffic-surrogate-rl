"""
SUMO-seed sweep: learned policies vs constant policies with error bars.

The Phase 1 scenario is fully deterministic (IDM sigma = 0, speedDev = 0,
fixed-rate mainline flow, deterministic ramp accumulator), so the SUMO seed
alone changes nothing (verified 2026-08-28: five seeds → byte-identical
trajectories). To obtain seed-to-seed variability this script enables
per-vehicle desired-speed heterogeneity via `--speed-dev` (SUMO `speedDev`;
SUMO's default is 0.1). The reward, network, demand and control step are
unchanged; routes are written to a separate network dir so training
artifacts are untouched.

Usage:
  python scripts/run_seed_sweep_sumo.py \\
      --policies u=0.5 u=0.55 u=0.6 runs/rl/<run>/best_model.zip \\
      --seeds 0 1 2 3 4 5 6 7 8 9 --speed-dev 0.1 \\
      --config configs/rl/ppo_sumo_m7_run4.yaml \\
      --out _progress/m7_seed_sweep_sd0.1.jsonl

Writes one JSON line per (policy, seed) episode and prints a summary table
(mean ± std, min, max, breakdown count where max mean density > 60 veh/km).
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
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_sumo_baselines import _load_sumo_env_config, rollout_policy_sumo  # noqa: E402
from rl.sumo_env_wrapper import SumoEnv  # noqa: E402

BREAKDOWN_DENSITY = 60.0  # veh/km mean over the corridor; free flow stays < 30


def summarize(rows: list[dict]) -> list[dict]:
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    out = []
    for policy, eps in by_policy.items():
        ret = np.array([e["total_reward"] for e in eps])
        qout = np.array([e["outflow_vph_mean"] for e in eps])
        umean = np.array([e["action_mean"] for e in eps])
        broke = int(sum(e["density_max"] > BREAKDOWN_DENSITY for e in eps))
        discarded = np.array([e.get("discarded_mainline", 0) for e in eps])
        pending_max = np.array([e.get("pending_mainline_max", 0) for e in eps])
        out.append({
            "policy": policy,
            "label": eps[0]["label"],
            "n": len(eps),
            "return_mean": float(ret.mean()),
            "return_std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
            "return_min": float(ret.min()),
            "return_max": float(ret.max()),
            "outflow_mean": float(qout.mean()),
            "action_mean": float(umean.mean()),
            "breakdowns": broke,
            "discarded_mainline_mean": float(discarded.mean()),
            "pending_mainline_max": int(pending_max.max()),
        })
    out.sort(key=lambda s: -s["return_mean"])
    return out


def print_summary(summary: list[dict], speed_dev: float, seeds: list[int]) -> None:
    print(f"\n== Seed sweep: speed_dev={speed_dev}, seeds={seeds} ==")
    print(f"{'policy':44s} {'n':>2} {'mean':>8} {'± std':>7} {'min':>8} {'max':>8} {'q_out':>6} {'u':>5} "
          f"{'breakdowns':>10} {'disc_main':>9} {'pend_max':>8}")
    for s in summary:
        print(f"{s['label'][:44]:44s} {s['n']:2d} {s['return_mean']:8.1f} {s['return_std']:7.1f} "
              f"{s['return_min']:8.1f} {s['return_max']:8.1f} {s['outflow_mean']:6.0f} {s['action_mean']:5.2f} "
              f"{s['breakdowns']:>6d}/{s['n']} {s['discarded_mainline_mean']:9.0f} {s['pending_mainline_max']:8d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO-seed sweep of policies with driver heterogeneity")
    parser.add_argument("--policies", nargs="+", required=True, help="u=<rate> specs and/or PPO .zip paths")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--speed-dev", type=float, default=0.1, help="SUMO vType speedDev (0 = deterministic)")
    parser.add_argument("--max-depart-delay", type=float, default=None,
                        help="override simulation.max_depart_delay_s (default: scenario config)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "rl" / "ppo_sumo.yaml")
    parser.add_argument("--network-dir", default="data/raw/rl_network_seed_sweep")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env_cfg = _load_sumo_env_config(overrides={}, config_path=args.config)
    env_cfg["network_dir"] = args.network_dir
    env_cfg["sumo_overrides"] = {"vehicle": {"speed_dev": float(args.speed_dev)}}
    if args.max_depart_delay is not None:
        env_cfg["sumo_overrides"]["simulation"] = {"max_depart_delay_s": float(args.max_depart_delay)}
    env = SumoEnv(env_cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        with args.out.open("w", encoding="utf-8") as f:
            for policy in args.policies:
                for seed in args.seeds:
                    r = rollout_policy_sumo(
                        policy_arg=policy, seed=seed, n_episodes=1,
                        config_path=args.config, env=env,
                    )
                    r["sumo_seed"] = seed
                    r["speed_dev"] = float(args.speed_dev)
                    r["max_depart_delay_s"] = float(r.get("max_depart_delay_s", -1.0))
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    if not args.quiet:
                        print(f"  {r['label'][:40]:40s} seed={seed:3d}  return={r['total_reward']:8.1f}  "
                              f"q_out={r['outflow_vph_mean']:5.0f}  rho_max={r['density_max']:5.0f}  "
                              f"u={r['action_mean']:.2f}  disc_main={r.get('discarded_mainline', 0):3d}")
    finally:
        env.close()

    summary = summarize(rows)
    print_summary(summary, args.speed_dev, args.seeds)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(rows)} episodes to {args.out} and summary to {summary_path}")


if __name__ == "__main__":
    main()
