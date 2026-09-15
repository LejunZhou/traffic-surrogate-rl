"""
Arms manifest for the M13 round-0 budget study (`_plans/m13_round0_budget_plan.md`).

  PYTHONPATH=src python scripts/build_m13_manifest.py --demo-arms runs/study/demo/arms.json \\
      --agg runs/aggregation/m13_mix1_s0 runs/aggregation/m13_mix2_s0 \\
      --names "A1 aggregation (N0=240, mix1)" "A1 aggregation (N0=240, mix2)" \\
      --ensembles runs/surrogate/plant_v2_r0s240_mix1 runs/surrogate/plant_v2_r0s240_mix2 \\
      --r0-studies m13_r0_mix1 m13_r0_mix2 --out runs/study/m13/arms.json

Reference arms ("B direct SUMO PPO", "ALINEA", "constant u") and the as-run
aggregation arm are copied from the demo manifest with their existing test /
OOD JSONL paths, so `run_final_evaluation.py` re-evaluates nothing there. Each
new aggregation study becomes a curve arm: a zero-shot point at the round-0
budget (A0_zero_shot.zip) and one point per round (selected_r{j}.zip) at the
round's cumulative EE; every point carries `ensemble_dir` so the evaluator
normalises densities with the statistics that policy was trained against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.ledger import Ledger  # noqa: E402

COPY_ARMS = ("B direct SUMO PPO", "ALINEA", "constant u")


def _ledger_wall(study: str) -> float:
    try:
        return float(sum(l["wall_s"] for l in Ledger(study, PROJECT_ROOT).lines() if l["purpose"] not in ("eval_test", "eval_ood")))
    except FileNotFoundError:
        return 0.0


def _ensemble_wall(ens: Path) -> float:
    wall = 0.0
    for log in ens.glob("member_*/metrics.csv"):
        try:
            wall = max(wall, float(log.read_text().strip().splitlines()[-1].split(",")[5]))
        except Exception:
            pass
    return wall


def _tag(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-arms", default="runs/study/demo/arms.json")
    ap.add_argument("--agg", nargs="+", required=True, help="aggregation study dirs (with rounds.json)")
    ap.add_argument("--names", nargs="*", default=None,
                    help="arm name per --agg entry (default: 'A1 aggregation (N0=<n0>, <mix>)' with <mix> from the study dir m13_<mix>_s<seed>)")
    ap.add_argument("--ensembles", nargs="+", required=True, help="round-0 ensemble dir per --agg entry")
    ap.add_argument("--r0-studies", nargs="+", required=True, help="ledger study of the round-0 dataset per --agg entry")
    ap.add_argument("--as-run-name", default="A1 aggregation (N0=692, as run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (len(args.agg) == len(args.ensembles) == len(args.r0_studies)):
        ap.error("--agg, --ensembles and --r0-studies must have the same length")
    if args.names and len(args.names) != len(args.agg):
        ap.error("--names must have one entry per --agg")
    out = PROJECT_ROOT / args.out
    ev = f"{out.parent.relative_to(PROJECT_ROOT)}/eval"
    arms = []
    dataset_ee = {}

    demo_path = PROJECT_ROOT / args.demo_arms
    if demo_path.exists():
        demo = json.loads(demo_path.read_text())
        by_name = {a["name"]: a for a in demo["arms"]}
        as_run = []
        if "A0 zero-shot" in by_name:
            p0 = dict(by_name["A0 zero-shot"]["points"][0]); p0["round"] = 0
            as_run.append(p0)
        if "A1 aggregation" in by_name:
            as_run += [dict(p) for p in by_name["A1 aggregation"]["points"]]
        if as_run:
            arms.append({"name": args.as_run_name, "kind": "curve", "points": as_run})
            dataset_ee[args.as_run_name] = int(demo.get("dataset_ee", as_run[0]["ee"]))
        for name in COPY_ARMS:
            if name in by_name:
                arms.append({"name": name, "kind": by_name[name].get("kind", "point"), "points": [dict(p) for p in by_name[name]["points"]]})
    else:
        print(f"[m13] demo manifest {demo_path} not found; reference arms omitted")

    names = args.names or [None] * len(args.agg)
    for agg, name, ens, r0_study in zip(args.agg, names, args.ensembles, args.r0_studies):
        agg_dir = PROJECT_ROOT / agg
        rounds_path = agg_dir / "rounds.json"
        if not rounds_path.exists():
            print(f"[m13] {rounds_path} missing; arm {name or agg} skipped")
            continue
        rounds = json.loads(rounds_path.read_text())
        n0 = int(rounds[0]["dataset_ee"])
        if name is None:
            parts = agg_dir.name.split("_")
            mix = parts[1] if len(parts) >= 3 and parts[0] == "m13" else agg_dir.name
            name = f"A1 aggregation (N0={n0}, {mix})"
        tag = _tag(name)
        base_wall = _ledger_wall(r0_study) + _ensemble_wall(PROJECT_ROOT / ens)
        pts = [{"ee": n0, "wall_s": base_wall + float(rounds[0]["ppo_wall_s"]), "seed": 0, "round": 0,
                "policy": str(agg_dir / "A0_zero_shot.zip"), "ensemble_dir": ens,
                "selection_ee": int(rounds[0]["aggregation_ee"]),
                "test": f"{ev}/{tag}_r0_test.jsonl", "ood": f"{ev}/{tag}_r0_ood.jsonl", "val": f"{ev}/{tag}_r0_val.jsonl"}]
        wall = base_wall
        for r in rounds:
            wall += float(r["wall_s"])
            j = int(r["round"])
            pts.append({"ee": int(r["cumulative_ee"]), "wall_s": wall, "seed": 0, "round": j,
                        "policy": str(agg_dir / f"selected_r{j}.zip"), "ensemble_dir": ens,
                        "test": f"{ev}/{tag}_r{j}_test.jsonl", "ood": f"{ev}/{tag}_r{j}_ood.jsonl", "val": f"{ev}/{tag}_r{j}_val.jsonl"})
        arms.append({"name": name, "kind": "curve", "points": pts})
        dataset_ee[name] = n0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"study": "m13", "dataset_ee": dataset_ee, "arms": arms}, indent=1))
    print(f"{len(arms)} arms -> {out}")
    for a in arms:
        print(f"  {a['name']:<34s} {len(a['points'])} point(s): " + ", ".join(f"{p['ee']} EE" for p in a["points"]))


if __name__ == "__main__":
    main()
