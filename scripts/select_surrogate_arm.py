"""
Select the policy of a surrogate-only PPO run (single-surrogate, one-step or
anticipative arm): rank checkpoints by the surrogate V return, roll the top 3
in SUMO on V (54 EE, ledger purpose aggregation), pick the best SUMO return
without catastrophic episodes -> <run>/best_model_selected.zip.

  PYTHONPATH=src python scripts/select_surrogate_arm.py --run runs/study/demo/single_surrogate --study demo_single
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.profile_eval import evaluate_on_profiles, load_set, print_summary, sumo_env_config  # noqa: E402
from run_aggregation_loop import rank_checkpoints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--study", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--catastrophic", type=float, default=-250.0)
    args = ap.parse_args()
    run = PROJECT_ROOT / args.run
    if (run / "best_model_selected.zip").exists() and (run / "eval" / "selection.json").exists():
        sel = json.loads((run / "eval" / "selection.json").read_text())["chosen"]
        print(f"{run.name}: already selected step {sel['step']} (SUMO {sel['sumo_mean']:.1f}); skipping")
        return
    top, _ = rank_checkpoints(run, args.top_k)
    profiles = load_set("val", PROJECT_ROOT)
    env_cfg = sumo_env_config(PROJECT_ROOT, network_dir=str(run / "network"),
                              extra={"ensemble_dir": args.ensemble, "observation": {"lookahead_steps": args.lookahead}})
    res = evaluate_on_profiles([c["path"] for c in top], profiles, env_cfg, run / "eval" / "sumo_val.jsonl", workers=args.workers,
                               purpose="aggregation", study=args.study, project_root=PROJECT_ROOT, network_root=str(run / "network"), quiet=True)
    print_summary(res["summary"], f"{run.name}: SUMO V of the top-{len(top)} checkpoints")
    pol = res["summary"]["policies"]
    for c in top:
        c["sumo_mean"] = pol[c["path"]]["mean"]; c["catastrophic"] = int(sum(1 for r in res["rows"] if r["policy"] == c["path"] and r["return"] < args.catastrophic))
    feasible = [c for c in top if c["catastrophic"] == 0] or top
    best = max(feasible, key=lambda c: c["sumo_mean"])
    shutil.copy(best["path"], run / "best_model_selected.zip")
    (run / "eval" / "selection.json").write_text(json.dumps({"chosen": best, "top": top}, indent=1))
    print(f"selected step {best['step']} (surrogate {best['surrogate_mean']:.1f}, SUMO {best['sumo_mean']:.1f}) -> best_model_selected.zip")


if __name__ == "__main__":
    main()
