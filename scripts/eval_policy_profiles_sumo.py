"""
Evaluate policies on a frozen profile set in SUMO (M11/M12, draft §10).

  PYTHONPATH=src python scripts/eval_policy_profiles_sumo.py \\
      --policies runs/rl/<run>/best_model.zip u=0.3 alinea:ki=15,rho=37,det=12 mpc:runs/surrogate/plant_v2_round0 \\
      --set test [--seeds 100 101 102] [--purpose eval_test] [--study m12_final] \\
      [--out runs/eval/final_test.jsonl] [--workers 8] [--reference <policy>]
      [--overlay configs/rl/env_sumo.yaml] [--lookahead 20]

Profile sets: val (18 x 1 seed), test (30 x 3), ood (12 x 3) or a JSON path.
Every episode is one ledger line; eval_test / eval_ood are not counted in
the SUMO budget. Writes <out>.jsonl (one line per episode) and
<out>.summary.json (per-policy metrics + paired bootstrap CIs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.profile_eval import evaluate_on_profiles, load_set, print_summary, sumo_env_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--set", default="val")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--study", default="eval")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/rl/ppo_common.yaml")
    ap.add_argument("--overlay", default="configs/rl/env_sumo.yaml")
    ap.add_argument("--lookahead", type=int, default=None, help="observation.lookahead_steps for anticipative policies")
    ap.add_argument("--ensemble-dir", default=None, help="take density stats from this ensemble (parity with the surrogate arm)")
    ap.add_argument("--save-rollouts", default=None, help="directory for the rollout npz files")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--network-dir", default="data/raw/rl_network_eval")
    args = ap.parse_args()

    profiles = load_set(args.set, PROJECT_ROOT)
    set_name = profiles[0].set_name if profiles else args.set
    purpose = args.purpose or {"val": "eval_val", "test": "eval_test", "ood": "eval_ood"}.get(set_name, "eval_val")
    extra = {}
    if args.lookahead is not None:
        extra["observation"] = {"lookahead_steps": int(args.lookahead)}
    if args.ensemble_dir:
        extra["ensemble_dir"] = args.ensemble_dir
    env_cfg = sumo_env_config(PROJECT_ROOT, args.config, args.overlay, network_dir=args.network_dir, extra=extra)
    out = Path(args.out or f"runs/eval/{args.study}_{set_name}.jsonl")
    out = out if out.is_absolute() else PROJECT_ROOT / out
    policies = list(args.policies)
    if args.reference and args.reference in policies:
        policies.remove(args.reference); policies.insert(0, args.reference)
    res = evaluate_on_profiles(policies, profiles, env_cfg, out, seeds=args.seeds, workers=args.workers, purpose=purpose,
                               study=args.study, round_index=args.round, project_root=PROJECT_ROOT,
                               save_rollouts_dir=None if args.save_rollouts is None else PROJECT_ROOT / args.save_rollouts,
                               network_root=str(PROJECT_ROOT / args.network_dir))
    print_summary(res["summary"], f"{set_name}: {len(profiles)} profiles x {len(args.seeds) if args.seeds else 'set'} seeds ({res['summary']['n_episodes']} episodes, {res['summary']['wall_s']:.0f} s)")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
