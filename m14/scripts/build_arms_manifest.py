"""Build the M14 evaluation manifest from generated checkpoints and ledgers.

Includes A0, A1, direct SUMO-PPO, tuned ALINEA/PI-ALINEA, and fixed constants.
A tuned constant is included when --constant is supplied. MPC is opt-in via
--mpc-spec. Runtime accounting is explicitly partial, not elapsed wall time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sumo_env.rollout_store import RolloutStore  # noqa: E402
from utils.ledger import Ledger  # noqa: E402


def _budget_rows(study: str, upto_round: int | None = None) -> list[dict]:
    return [r for r in Ledger(study, PROJECT_ROOT).lines()
            if r["purpose"] not in ("eval_test", "eval_ood")
            and (upto_round is None or r["round"] <= upto_round)]


def _runtime(rows: list[dict]) -> float:
    return sum(float(r.get("wall_s", 0.0)) for r in rows)


def _ensemble_runtime(directory: Path) -> float:
    """Sum recorded member training durations; parallel time is counted per member."""
    total = 0.0
    for path in directory.glob("member_*/metrics.csv"):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            total += float(rows[-1].get("wall_s", 0.0))
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="m14")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--direct-ee", type=int, nargs="*", default=[200, 700])
    ap.add_argument("--alinea", required=True, help="selected alinea: or pialinea: controller specification")
    ap.add_argument("--constant", default=None, help="optional validation-selected constant specification, e.g. u=0.5")
    ap.add_argument("--constants", type=float, nargs="*", default=[0.0, 0.5, 1.0], help="fixed, untuned reference rates")
    ap.add_argument("--ensemble", default="runs/deeponet/round0")
    ap.add_argument("--store", default="data/round0")
    ap.add_argument("--mpc-spec", default=None, help="optional mpc:<ensemble>,iters=30 controller")
    ap.add_argument("--no-mpc", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="runs/study/m14/arms.json")
    ap.add_argument("--r0-studies", nargs="*", default=["m14_round0", "m14_e0"])
    args = ap.parse_args()
    if any(not 0 <= u <= 1 for u in args.constants):
        ap.error("--constants values must be in [0, 1]")
    out = (PROJECT_ROOT / args.out).resolve()
    ev = out.parent / "eval"
    try:
        ev = ev.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    dataset_ee = sum(e["round"] == 0 for e in RolloutStore(PROJECT_ROOT / args.store).entries)
    base_runtime = _ensemble_runtime(PROJECT_ROOT / args.ensemble)
    base_runtime += sum(_runtime(_budget_rows(st)) for st in args.r0_studies)
    arms = []

    def point(key: str, policy: str, ee: int, runtime: float, seed: int = 0, **extra) -> dict:
        return {"ee": ee, "accounted_runtime_s": runtime, "seed": seed, "policy": policy,
                "test": str(ev / f"{key}_test.jsonl"), "ood": str(ev / f"{key}_ood.jsonl"), **extra}

    def add(name: str, kind: str, points: list[dict]) -> None:
        if points:
            arms.append({"name": name, "kind": kind, "points": points})

    a0, a1 = [], []
    for seed in args.seeds:
        agg = PROJECT_ROOT / f"runs/aggregation/{args.study}_s{seed}"
        if not (agg / "rounds.json").exists():
            continue
        rounds = json.loads((agg / "rounds.json").read_text())
        if not rounds:
            continue
        selected_round = max(rounds, key=lambda row: row["best_sumo_val"])["round"]
        learning_runtime = base_runtime
        seen_ensembles = {(PROJECT_ROOT / args.ensemble).resolve()}
        for row in rounds:
            learning_runtime += float(row["ppo_wall_s"])
            after = Path(row["ensemble_after"])
            if not after.is_absolute():
                after = PROJECT_ROOT / after
            if after.resolve() not in seen_ensembles:
                learning_runtime += _ensemble_runtime(after)
                seen_ensembles.add(after.resolve())
            runtime = learning_runtime + _runtime(_budget_rows(f"{args.study}_s{seed}", row["round"]))
            a1.append(point(f"A1_r{row['round']}_s{seed}", str(agg / f"selected_r{row['round']}.zip"),
                            row["cumulative_ee"], runtime, seed, round=row["round"], validation_selected=row["round"] == selected_round,
                            ensemble_dir=str(row["ensemble_before"])))
        a0.append(point(f"A0_s{seed}", str(agg / "A0_zero_shot.zip"), dataset_ee,
                        base_runtime + float(rounds[0]["ppo_wall_s"]), seed,
                        ensemble_dir=str(PROJECT_ROOT / args.ensemble)))
    add("A0 zero-shot", "point", a0)
    add("A1 aggregation", "curve", a1)

    direct = []
    for ee in args.direct_ee:
        for seed in args.seeds:
            checkpoint = PROJECT_ROOT / f"runs/study/{args.study}/direct_ppo_{ee}ee_s{seed}/best_model_selected.zip"
            if checkpoint.exists():
                rows = _budget_rows(f"{args.study}_direct_{ee}_s{seed}")
                direct.append(point(f"B_{ee}_s{seed}", str(checkpoint), len(rows), _runtime(rows), seed, nominal_ee=ee))
    for seed in args.seeds:
        points = [p for p in direct if p["seed"] == seed]
        if points:
            largest_budget = max(p["nominal_ee"] for p in points)
            for p in points:
                p["budget_selected"] = p["nominal_ee"] == largest_budget
    add("B direct SUMO PPO", "curve", direct)

    tuning = _budget_rows(f"{args.study}_alinea")
    feedback_rows = [r for r in tuning if str(r["policy"]).startswith(("alinea:", "pialinea:"))]
    constant_rows = [r for r in tuning if str(r["policy"]).startswith("u=")]
    add("ALINEA / PI-ALINEA", "band", [point("alinea", args.alinea, len(feedback_rows), _runtime(feedback_rows))])
    if args.constant:
        add("tuned constant u", "band", [point("constant", args.constant, len(constant_rows), _runtime(constant_rows))])
    for u in args.constants:
        label = f"u={u:g}"
        add(f"fixed {label}", "band", [point(f"constant_fixed_{u:g}", label, 0, 0.0)])
    if args.mpc_spec and not args.no_mpc:
        add("Surrogate-MPC", "point", [point("mpc", args.mpc_spec, dataset_ee, base_runtime,
                                             ensemble_dir=str(PROJECT_ROOT / args.ensemble))])
    payload = {
        "study": args.study, "dataset_ee": dataset_ee, "arms": arms,
        "runtime_accounting_note": "accounted_runtime_s is partial accounting, not elapsed wall-clock or normalized compute. "
            "It sums available SUMO episode durations and, for surrogate arms, member-training and PPO-process durations. "
            "Parallel work is counted per episode/member; direct-PPO optimization overhead is not in its episode ledger. "
            "Use runs/commands.jsonl for measured command elapsed times.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"{len(arms)} arms -> {out}")
    for arm in arms:
        print(f"  {arm['name']}: " + ", ".join(f"{p['ee']} EE" for p in arm["points"]))


if __name__ == "__main__":
    main()
