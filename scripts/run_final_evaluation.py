"""
E8 — final evaluation of every arm on the test set T (30 profiles x 3 seeds)
and the OOD set O (12 x 3) (draft §10). Not counted in the SUMO budget.

  PYTHONPATH=src python scripts/run_final_evaluation.py --arms runs/study/demo/arms.json --workers 8 --study demo_final
Skips (arm, set) pairs whose JSONL already exists; rollouts of the test set
are saved next to the JSONL for the policy-structure figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.profile_eval import evaluate_on_profiles, load_set, print_summary, sumo_env_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--study", default="final")
    ap.add_argument("--sets", nargs="*", default=["test", "ood"])
    ap.add_argument("--only", nargs="*", default=None, help="arm names to evaluate")
    args = ap.parse_args()
    manifest = json.loads((PROJECT_ROOT / args.arms).read_text())
    ens_dir = None
    for arm in manifest["arms"]:
        for p in arm["points"]:
            if str(p["policy"]).startswith("mpc:"):
                ens_dir = p["policy"][4:]
    for set_name in args.sets:
        profiles = load_set(set_name, PROJECT_ROOT)
        purpose = {"test": "eval_test", "ood": "eval_ood", "val": "eval_val"}[set_name]
        for arm in manifest["arms"]:
            if args.only and arm["name"] not in args.only:
                continue
            for p in arm["points"]:
                out = PROJECT_ROOT / p[set_name]
                if out.exists():
                    continue
                extra = {"observation": {"lookahead_steps": int(p.get("lookahead", 0))}}
                # density normalisation comes from the ensemble the policy was trained against
                # (per-point `ensemble_dir`, e.g. the M13 arms); else the study's MPC ensemble; else ppo_common's stats file
                if p.get("ensemble_dir"):
                    extra["ensemble_dir"] = p["ensemble_dir"]
                elif ens_dir:
                    extra["ensemble_dir"] = ens_dir
                env_cfg = sumo_env_config(PROJECT_ROOT, network_dir=f"data/raw/rl_network_final_{set_name}", extra=extra)
                print(f"[final] {arm['name']} @ {p['ee']} EE: density stats {env_cfg['density_mean']:.3f} / {env_cfg['density_std']:.3f} "
                      f"from {extra.get('ensemble_dir', 'density_stats_from')}", flush=True)
                res = evaluate_on_profiles([p["policy"]], profiles, env_cfg, out, workers=args.workers, purpose=purpose, study=args.study,
                                           project_root=PROJECT_ROOT, network_root=f"{PROJECT_ROOT}/data/raw/rl_network_final_{set_name}",
                                           save_rollouts_dir=out.parent / "rollouts" / out.stem if set_name == "test" else None,
                                           labels={p["policy"]: arm["name"]}, quiet=True)
                print_summary(res["summary"], f"{arm['name']} @ {p['ee']} EE on {set_name} ({res['summary']['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
