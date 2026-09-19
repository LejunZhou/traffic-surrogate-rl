"""
Tune ALINEA/PI-ALINEA and a constant meter on validation profiles.

Stage 1 compares detector/set-point/integral-gain combinations on a subset;
stage 2 evaluates the best combinations and proportional variants on full
validation. An optional constant sweep uses u=0,0.1,...,1. Selection never
uses test or OOD outcomes. All tuning episodes enter the shared study ledger.
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--study", default="m14_alinea")
    ap.add_argument("--stage1-profiles", type=int, nargs="*", default=[1, 4, 7, 10, 13, 16])
    ap.add_argument("--dets", type=int, nargs="*", default=[11, 12, 13])
    ap.add_argument("--rhos", type=float, nargs="*", default=[26, 30, 34, 38])
    ap.add_argument("--kis", type=float, nargs="*", default=[10, 20, 35])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--skip-constants", action="store_true")
    ap.add_argument("--out", default="runs/study/m14/alinea_tuning.json")
    args = ap.parse_args()
    val = load_set("val", PROJECT_ROOT)
    env_cfg = sumo_env_config(PROJECT_ROOT, network_dir="data/networks/tune")
    out_dir = PROJECT_ROOT / "runs" / "eval" / args.study
    report = {}
    # stage 1
    subset = [val[i] for i in args.stage1_profiles]
    cands = [f"alinea:ki={ki:g},rho={rho:g},det={det}" for det in args.dets for rho in args.rhos for ki in args.kis]
    res1 = evaluate_on_profiles(cands, subset, env_cfg, out_dir / "stage1.jsonl", workers=args.workers, purpose="tuning",
                                study=args.study, project_root=PROJECT_ROOT, network_root=str(PROJECT_ROOT / "data/networks/tune"), quiet=True)
    print_summary(res1["summary"], f"stage 1: {len(cands)} candidates x {len(subset)} profiles")
    pol1 = res1["summary"]["policies"]
    ranked = sorted(pol1, key=lambda p: (pol1[p]["n_catastrophic"] > 0, -pol1[p]["mean"]))[: args.top]
    report["stage1"] = {p: pol1[p] for p in pol1}
    # stage 2: top candidates + PI variants on all of V
    cands2 = list(ranked) + [c.replace("alinea:", "pialinea:kp=4,") for c in ranked[:2]]
    res2 = evaluate_on_profiles(cands2, val, env_cfg, out_dir / "stage2.jsonl", workers=args.workers, purpose="tuning",
                                study=args.study, project_root=PROJECT_ROOT, network_root=str(PROJECT_ROOT / "data/networks/tune"), quiet=True)
    print_summary(res2["summary"], f"stage 2: {len(cands2)} candidates x 18 V profiles")
    pol2 = res2["summary"]["policies"]
    best = min(pol2, key=lambda p: (pol2[p]["n_catastrophic"] > 0, -pol2[p]["mean"]))
    report["stage2"] = {p: pol2[p] for p in pol2}
    report["best_alinea"] = best
    # constants
    if not args.skip_constants:
        consts = [f"u={0.1 * i:.1f}" for i in range(11)]
        res3 = evaluate_on_profiles(consts, val, env_cfg, out_dir / "constants.jsonl", workers=args.workers, purpose="tuning",
                                    study=args.study, project_root=PROJECT_ROOT, network_root=str(PROJECT_ROOT / "data/networks/tune"), quiet=True)
        print_summary(res3["summary"], "constant u on V")
        pol3 = res3["summary"]["policies"]
        report["constants"] = {p: pol3[p] for p in pol3}
        report["best_constant"] = max(pol3, key=lambda p: pol3[p]["mean"])
    out = PROJECT_ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nbest ALINEA on V: {best} ({pol2[best]['mean']:.1f}); best constant: {report.get('best_constant')}; written {out}")


if __name__ == "__main__":
    main()
