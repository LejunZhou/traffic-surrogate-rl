"""
Constant-u sweep in live SUMO (Milestone 7 step 1).

Rolls constant ramp-metering policies u ∈ {0.0, 0.1, ..., 1.0} through
SumoEnv once each and writes one JSON line per u with the per-step arrays
(outflow, queue, density std, ...) that scripts/balance_reward_terms.py
needs to (a) pin q_ref to the measured outflow peak, (b) check whether a
capacity drop exists (q_out peaking at some u* < 1), and (c) balance the
three reward weights offline.

Reward weights during the sweep do not matter for the recorded arrays; the
run uses the config's weights so `total_reward` is also directly comparable
to later PPO evals.

Usage:
  python scripts/run_u_sweep_sumo.py --out _progress/m7_u_sweep_seed0.jsonl
  python scripts/run_u_sweep_sumo.py --u 0 0.25 0.5 0.75 1 --seed 1 --out sweep.jsonl

Wall clock: ~13 s per point on the M6 machine → ~2.5 min for 11 points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_sumo_baselines import (  # noqa: E402
    _load_sumo_env_config,
    print_single_episode,
    rollout_policy_sumo,
)
from rl.sumo_env_wrapper import SumoEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Constant-u sweep in SumoEnv")
    parser.add_argument(
        "--u",
        type=float,
        nargs="+",
        default=[round(0.1 * i, 1) for i in range(11)],
        help="Constant metering rates to evaluate (default 0.0..1.0 step 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demands", type=float, nargs="*", default=None,
                        help="mainline demand levels (vph); default: the config's demand_levels sampling")
    parser.add_argument("--ramp-demands", type=float, nargs="*", default=None,
                        help="ramp arrival rates (vph); default: the config's ramp_demand_levels sampling")
    parser.add_argument("--network-dir", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="PPO config whose env block to use (default configs/rl/ppo_sumo.yaml)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env_cfg = _load_sumo_env_config(overrides={}, config_path=args.config)
    if args.demands:
        env_cfg["demand_levels"] = [float(d) for d in args.demands]
    if args.ramp_demands:
        env_cfg["ramp_demand_levels"] = [float(r) for r in args.ramp_demands]
    if args.network_dir:
        env_cfg["network_dir"] = args.network_dir
    env = SumoEnv(env_cfg)
    cells = [(d, r) for d in (args.demands or [None]) for r in (args.ramp_demands or [None])]
    n = 0
    try:
        with args.out.open("w", encoding="utf-8") as f:
            for demand, ramp in cells:
                options = {}
                if demand is not None:
                    options["demand_vph"] = float(demand)
                if ramp is not None:
                    options["ramp_demand_vph"] = float(ramp)
                for u in args.u:
                    result = rollout_policy_sumo(
                        policy_arg=f"u={u}",
                        seed=args.seed,
                        n_episodes=1,
                        config_path=args.config,
                        env=env,
                        reset_options=options or None,
                    )
                    result["u"] = float(u)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    n += 1
                    if not args.quiet:
                        print_single_episode(result, args.seed, 1)
                    else:
                        print(f"  demand={result['demand_vph']:.0f} ramp={result['ramp_demand_vph']:.0f} u={u:.1f} "
                              f"return={result['total_reward']:8.1f} q_out={result['outflow_vph_mean']:5.0f} "
                              f"rho_max={result['density_max']:4.0f} queue_final={result['queue_final']:4.0f}", flush=True)
    finally:
        env.close()
    print(f"wrote {n} sweep points to {args.out}")


if __name__ == "__main__":
    main()
