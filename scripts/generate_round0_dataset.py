"""
Generate the round-0 plant-model dataset from the behaviour-controller
mixture (M8 step 2, draft_pipeline.md §6.1) and write the rollout store.

  PYTHONPATH=src python scripts/generate_round0_dataset.py \\
      --config configs/experiments/round0_mixture.yaml [--n 480] [--workers 8]
      [--target-total 480]   # count rollouts already in the store (e.g. the E0
                             # sweeps) towards the mixture shares
      [--study m9_round0]    # ledger study name (purpose = dataset)
      [--store-dir ... --network-dir ...]   # override output paths (smoke runs)

Each rollout: a random training-family profile, a random SUMO seed,
speed_dev 0.03, one behaviour controller. Existing files in the store are
kept; only missing plan entries are generated (append-only, reproducible).

M13 keys under `dataset`: `per_entry_seeds` (seed every controller draw
independently), `storage_mandatory_frac` / `storage_mandatory_vph` (re-draw
that fraction of the profiles until the peak total demand exceeds the merge
capacity). Note: `--target-total` counts `feedforward` separately from the
`store_flush` share it is folded from in legacy configs.
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
    ap.add_argument("--config", default="configs/experiments/round0_mixture.yaml")
    ap.add_argument("--n", type=int, default=None, help="override dataset.n_rollouts")
    ap.add_argument("--target-total", type=int, default=None,
                    help="generate only enough new rollouts so the store reaches this size, honouring the shares")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--study", default="m9_round0")
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--store-dir", default=None, help="override output.store_dir")
    ap.add_argument("--network-dir", default=None, help="override output.network_dir")
    args = ap.parse_args()

    cfg = load_config(str(PROJECT_ROOT / args.config))
    ds = cfg["dataset"]; out = cfg["output"]
    env_cfg = dict(cfg["env"]); env_cfg["project_root"] = str(PROJECT_ROOT)
    store = RolloutStore(PROJECT_ROOT / (args.store_dir or out["store_dir"]))
    network_dir = args.network_dir or out.get("network_dir", "data/plant_v2/network")
    n = int(args.n or ds["n_rollouts"])
    shares = dict(ds.get("mixture", DEFAULT_SHARES))
    legacy = ds.get("legacy_policy_path")
    legacy_path = str(PROJECT_ROOT / legacy) if legacy else None

    if args.target_total is not None:
        # subtract what the store already holds per controller type
        have = store.summary()["by_controller"]
        total = sum(shares.values())
        want = {k: int(round(args.target_total * v / total)) for k, v in shares.items()}
        if legacy_path is None or not Path(legacy_path).exists():
            want["alinea"] = want.get("alinea", 0) + want.pop("legacy_policy", 0)
        need = {k: max(0, want[k] - have.get(k, 0)) for k in want}
        n = sum(need.values())
        shares = {k: float(v) for k, v in need.items() if v > 0}
        print(f"[round0] store has {store.summary()['n']} rollouts {have}; generating {n} more: {need}")
    plan = build_mixture_plan(n, int(ds.get("seed", 0)), shares, round_index=0,
                              legacy_policy_path=legacy_path, sumo_seed_base=int(ds.get("sumo_seed_base", 50000)),
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
