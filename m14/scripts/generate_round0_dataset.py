"""
Generate the M14 round-0 dataset from six behavior-controller families.

Existing rollout files are preserved; only missing planned trajectories run.
Use --n to set the new mixture size, or --target-total to count existing E0
rollouts toward a requested total. --store-dir and --network-dir isolate runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.behaviour_controllers import DEFAULT_SHARES, build_mixture_plan, enforce_storage_mandatory  # noqa: E402
from sumo_env.demand_profiles import ProfileFamily  # noqa: E402
from sumo_env.parallel_rollouts import run_jobs  # noqa: E402
from sumo_env.rollout_store import RolloutStore  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.ledger import Ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--n", type=int, default=None, help="override dataset.n_rollouts")
    ap.add_argument("--target-total", type=int, default=None,
                    help="generate only enough new rollouts so the store reaches this size, honouring the shares")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--study", default="m14_round0")
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--store-dir", default=None, help="override output.store_dir")
    ap.add_argument("--network-dir", default=None, help="override output.network_dir")
    args = ap.parse_args()

    cfg = load_config(str(PROJECT_ROOT / args.config))
    ds = cfg["dataset"]; out = cfg["output"]
    env_cfg = dict(cfg["env"]); env_cfg["project_root"] = str(PROJECT_ROOT)
    store = RolloutStore(PROJECT_ROOT / (args.store_dir or out["store_dir"]))
    network_dir = args.network_dir or out.get("network_dir", "data/networks/dataset")
    n = int(args.n or ds["n_rollouts"])
    shares = dict(ds.get("mixture", DEFAULT_SHARES))

    if args.target_total is not None:
        # subtract what the store already holds per controller type
        have = store.summary()["by_controller"]
        total = sum(shares.values())
        want = {k: int(round(args.target_total * v / total)) for k, v in shares.items()}
        need = {k: max(0, want[k] - have.get(k, 0)) for k in want}
        n = sum(need.values())
        shares = {k: float(v) for k, v in need.items() if v > 0}
        print(f"[round0] store has {store.summary()['n']} rollouts {have}; generating {n} more: {need}")
    plan = build_mixture_plan(n, int(ds.get("seed", 0)), shares, round_index=0,
                              sumo_seed_base=int(ds.get("sumo_seed_base", 50000)),
                              per_entry_seeds=bool(ds.get("per_entry_seeds", False)),
                              feedforward_capacity_vph=ds.get("feedforward_capacity_vph"))
    frac = float(ds.get("storage_mandatory_frac", 0.0))
    if frac > 0:
        family = ProfileFamily.load(PROJECT_ROOT / cfg["env"]["profiles"]["family"])
        n_lanes = int(load_config(str(PROJECT_ROOT / cfg["env"]["sumo_config"])).get("network", {}).get("num_lanes", 1))
        summary = enforce_storage_mandatory(plan, family, frac, float(ds.get("storage_mandatory_vph", 2500.0)), n_lanes=n_lanes)
        print(f"[round0] storage-mandatory profiles: {summary}")
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / "generation_plan_summary.json").write_text(json.dumps(summary, indent=1))
    existing = {e["file"] for e in store.entries}
    jobs = []
    for p in plan:
        name = f"r0_{p['controller']['type']}_{p['index']:05d}"
        if f"{name}.npz" in existing:
            continue
        jobs.append({**p, "name": name, "purpose": "dataset", "study": args.study,
                     "profile": {"set": "train", "draw": p["profile_draw"]}, "out_dir": str(store.root)})
    print(f"[round0] plan {len(plan)} rollouts, {len(jobs)} to generate, workers={args.workers or out.get('workers', 8)}")
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "generation_plan.json").write_text(json.dumps(plan, indent=1))
    t0 = time.time()
    results = run_jobs(jobs, env_cfg, workers=int(args.workers or out.get("workers", 8)),
                       network_root=str(PROJECT_ROOT / network_dir))
    ledger = Ledger(args.study, PROJECT_ROOT)
    for r in results:
        store.add(r["path"], r["meta"])
        ledger.log(round_index=0, purpose="dataset", profile_set="train", profile_id=r["meta"]["profile_draw"],
                   sumo_seed=r["meta"]["sumo_seed"], policy=r["meta"]["controller"]["type"],
                   ret=r["metrics"]["return"], breakdown=r["metrics"]["breakdown"], wall_s=r["metrics"]["wall_s"])
    store.save_index()
    if not args.no_splits:
        sp = cfg.get("splits", {})
        store.make_splits(float(sp.get("train_frac", 0.7)), float(sp.get("val_frac", 0.15)), int(sp.get("seed", 0)))
    print(f"[round0] done in {time.time() - t0:.0f} s; store: {json.dumps(store.summary())}")


if __name__ == "__main__":
    main()
