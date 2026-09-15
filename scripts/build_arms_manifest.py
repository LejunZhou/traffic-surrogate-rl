"""
Collect every arm of the study into arms.json for the final evaluation and
the figures (draft §8): policy spec, cumulative SUMO budget (EE) from the
ledgers, wall-clock, seed, and the evaluation file paths to be filled by
scripts/run_final_evaluation.py.

  PYTHONPATH=src python scripts/build_arms_manifest.py --study demo --seeds 0 --direct-ee 200 700 --ft-ee 100 \\
      --alinea "alinea:ki=20,rho=34,det=12" --constant u=0.5 --out runs/study/demo/arms.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sumo_env.rollout_store import RolloutStore  # noqa: E402
from utils.ledger import Ledger  # noqa: E402


def _ledger_budget(study: str) -> int:
    return Ledger(study, PROJECT_ROOT).budget()


def _ledger_wall(study: str) -> float:
    return float(sum(l["wall_s"] for l in Ledger(study, PROJECT_ROOT).lines() if l["purpose"] not in ("eval_test", "eval_ood")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--direct-ee", type=int, nargs="*", default=[200, 700])
    ap.add_argument("--ft-ee", type=int, default=100)
    ap.add_argument("--alinea", required=True)
    ap.add_argument("--constant", required=True)
    ap.add_argument("--ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--store", default="data/plant_v2/round0")
    ap.add_argument("--mpc-spec", default=None, help="default mpc:<ensemble>,iters=30 (demo: iters=20,members=0-2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = (PROJECT_ROOT / args.out).parent
    ev = f"{out_dir.relative_to(PROJECT_ROOT)}/eval"
    dataset_ee = sum(1 for e in RolloutStore(PROJECT_ROOT / args.store).entries if e["round"] == 0)
    ens_wall = 0.0
    for log in (PROJECT_ROOT / args.ensemble).glob("member_*/metrics.csv"):
        try:
            ens_wall = max(ens_wall, float(log.read_text().strip().splitlines()[-1].split(",")[5]))
        except Exception:
            pass
    ens_wall += _ledger_wall("m9_round0") + _ledger_wall("m8_e0")
    arms = []

    def add(name, kind, points):
        pts = [p for p in points if p is not None]
        if pts:
            arms.append({"name": name, "kind": kind, "points": pts})

    a0, a1, a2 = [], [], []
    for s in args.seeds:
        agg = PROJECT_ROOT / f"runs/aggregation/{args.study}_s{s}"
        if not (agg / "rounds.json").exists():
            continue
        rounds = json.loads((agg / "rounds.json").read_text())
        wall = ens_wall
        for r in rounds:
            wall += r["wall_s"]
            key = f"A1_r{r['round']}_s{s}"
            a1.append({"ee": r["cumulative_ee"], "wall_s": wall, "seed": s, "round": r["round"], "policy": str(agg / f"selected_r{r['round']}.zip"),
                       "test": f"{ev}/{key}_test.jsonl", "ood": f"{ev}/{key}_ood.jsonl"})
        a0.append({"ee": dataset_ee, "wall_s": ens_wall + rounds[0]["ppo_wall_s"], "seed": s, "policy": str(agg / "A0_zero_shot.zip"),
                   "selection_ee": rounds[0]["aggregation_ee"], "test": f"{ev}/A0_s{s}_test.jsonl", "ood": f"{ev}/A0_s{s}_ood.jsonl"})
    add("A0 zero-shot", "point", a0)
    add("A1 aggregation", "curve", a1)
    ft = PROJECT_ROOT / f"runs/study/{args.study}/finetune_a2_{args.ft_ee}ee"
    if (ft / "best_model_selected.zip").exists() and a1:
        ee = a1[0]["ee"] + _ledger_budget(f"{args.study}_finetune")
        a2.append({"ee": ee, "wall_s": a1[0]["wall_s"] + _ledger_wall(f"{args.study}_finetune"), "seed": args.seeds[0],
                   "policy": str(ft / "best_model_selected.zip"), "test": f"{ev}/A2_test.jsonl", "ood": f"{ev}/A2_ood.jsonl"})
    add("A2 pre-train + fine-tune", "point", a2)
    b = []
    for ee in args.direct_ee:
        for s in args.seeds:
            run = PROJECT_ROOT / f"runs/study/{args.study}/direct_ppo_{ee}ee_s{s}"
            if (run / "best_model_selected.zip").exists():
                st = f"{args.study}_direct_{ee}_s{s}"
                b.append({"ee": _ledger_budget(st), "wall_s": _ledger_wall(st), "seed": s, "nominal_ee": ee,
                          "policy": str(run / "best_model_selected.zip"), "test": f"{ev}/B_{ee}_s{s}_test.jsonl", "ood": f"{ev}/B_{ee}_s{s}_ood.jsonl"})
    add("B direct SUMO PPO", "curve", b)
    add("ALINEA", "band", [{"ee": _ledger_budget(f"{args.study}_alinea"), "wall_s": _ledger_wall(f"{args.study}_alinea"), "seed": 0,
                            "policy": args.alinea, "test": f"{ev}/alinea_test.jsonl", "ood": f"{ev}/alinea_ood.jsonl"}])
    add("constant u", "band", [{"ee": 198, "wall_s": 0.0, "seed": 0, "policy": args.constant,
                                "test": f"{ev}/constant_test.jsonl", "ood": f"{ev}/constant_ood.jsonl"}])
    add("Surrogate-MPC", "point", [{"ee": dataset_ee, "wall_s": ens_wall, "seed": 0, "policy": args.mpc_spec or f"mpc:{args.ensemble},iters=30",
                                    "test": f"{ev}/mpc_test.jsonl", "ood": f"{ev}/mpc_ood.jsonl"}])
    for key, name in (("single_surrogate", "single-surrogate"), ("onestep_arm", "one-step model"), ("anticipative", "anticipative")):
        run = PROJECT_ROOT / f"runs/study/{args.study}/{key}"
        if (run / "best_model_selected.zip").exists():
            st = {"single_surrogate": f"{args.study}_single", "onestep_arm": f"{args.study}_onestep", "anticipative": f"{args.study}_anticipative"}[key]
            add(name, "point", [{"ee": dataset_ee + _ledger_budget(st), "wall_s": ens_wall, "seed": args.seeds[0],
                                 "policy": str(run / "best_model_selected.zip"), "lookahead": 20 if key == "anticipative" else 0,
                                 "test": f"{ev}/{key}_test.jsonl", "ood": f"{ev}/{key}_ood.jsonl"}])
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"study": args.study, "dataset_ee": dataset_ee, "arms": arms}, indent=1))
    print(f"{len(arms)} arms -> {out}")
    for a in arms:
        print(f"  {a['name']:<26s} {len(a['points'])} point(s): " + ", ".join(f"{p['ee']} EE" for p in a["points"]))


if __name__ == "__main__":
    main()
