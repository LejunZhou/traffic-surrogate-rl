"""Parallel SUMO rollout generation for the family-schema dataset config.

Runs the (otherwise serial) `sumo_env.dataset_generation` across N worker
processes safely:

1. Build the network + routes + detectors ONCE in the launcher (fixes the
   per-process network-rebuild race; run_simulation is read-only on them).
2. Build the deterministic generation plan; compute the MISSING rollouts
   (so a pilot run extends into a full run, append-only, reusing existing files).
3. Assign each worker a balanced (strided) subset of missing global indices and
   spawn it with `--reuse-network --no-splits --indices-file <json>`.
4. Wait for all workers; teleport QA gate; run make_splits ONCE (stratified).

Because each rollout's RNG is addressed by (base_seed, family_id, local_index),
the slicing is irrelevant to content — any partition reproduces the same data.

Usage:
    PYTHONPATH=src python scripts/run_parallel_generation.py \
        --config configs/experiments/dataset_2000vph_shockwave.yaml --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.config import load_config  # noqa: E402
from sumo_env.dataset_generation import (  # noqa: E402
    build_generation_plan,
    _resolve_network_files,
    make_splits,
)


def _npz_name(spec) -> str:
    return f"{spec.name}_{spec.local_index:05d}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="family-schema dataset config YAML")
    ap.add_argument("--workers", type=int, default=8, help="number of parallel SUMO workers")
    ap.add_argument("--overwrite", action="store_true", help="regenerate even if files exist")
    ap.add_argument(
        "--max-teleports", type=int, default=5,
        help="QA gate: quarantine rollouts whose teleport count exceeds this",
    )
    args = ap.parse_args()

    config_path = (_ROOT / args.config).resolve()
    ds_cfg = load_config(str(config_path))
    base_sumo_config = load_config(str(_ROOT / ds_cfg["base_sumo_config"]))
    ds, out = ds_cfg["dataset"], ds_cfg["output"]
    if "families" not in ds:
        ap.error("run_parallel_generation requires a family-schema config (dataset.families).")

    raw_dir = _ROOT / out["raw_dir"]
    network_dir = _ROOT / out["network_dir"]
    splits_dir = _ROOT / out["splits_dir"]
    demand_levels = ds.get("demand_levels", [2000])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pre-build network/routes/detectors once.
    print(f"[parallel] Pre-building network in {network_dir} ...")
    _resolve_network_files(network_dir, base_sumo_config, demand_levels, reuse_network=False)

    # 2) Plan + missing work.
    plan = build_generation_plan(ds["families"])
    N = len(plan)
    plan_path = raw_dir / "generation_plan.json"
    with open(plan_path, "w") as f:
        json.dump(
            [{"global_index": g, "family_id": s.family_id, "name": s.name,
              "local_index": s.local_index, "count": s.count} for g, s in enumerate(plan)],
            f, indent=2,
        )
    missing = [
        g for g, s in enumerate(plan)
        if args.overwrite or not (raw_dir / _npz_name(s)).exists()
    ]
    print(f"[parallel] Plan: {N} rollouts; {N - len(missing)} already present; "
          f"{len(missing)} to generate across {args.workers} workers.")

    if missing:
        n_workers = max(1, min(args.workers, len(missing)))
        worker_dir = raw_dir / "_worker_indices"
        worker_dir.mkdir(exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(_ROOT / "src")}

        procs = []
        for w in range(n_workers):
            idx = missing[w::n_workers]  # strided → balanced, even if pilot is done
            if not idx:
                continue
            idx_file = worker_dir / f"worker_{w}.json"
            idx_file.write_text(json.dumps(idx))
            cmd = [
                sys.executable, "-m", "sumo_env.dataset_generation",
                "--config", str(config_path), "--reuse-network", "--no-splits",
                "--indices-file", str(idx_file),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            log = open(worker_dir / f"worker_{w}.log", "w")
            procs.append((w, subprocess.Popen(cmd, cwd=str(_ROOT), env=env,
                                              stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[parallel]   worker {w}: {len(idx)} rollouts → {idx_file.name}")

        failed = []
        for w, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"[parallel] worker {w} exited with code {rc}")
            if rc != 0:
                failed.append(w)
        if failed:
            raise SystemExit(
                f"[parallel] workers {failed} failed; see {worker_dir}/worker_*.log"
            )

    # 3) Teleport QA gate.
    quarantine = raw_dir / "_quarantine"
    flagged = []
    for npz in sorted(raw_dir.glob("*.npz")):
        try:
            tel = int(np.load(npz)["teleports"])
        except (KeyError, Exception):
            continue
        if tel > args.max_teleports:
            flagged.append((npz.name, tel))
    if flagged:
        quarantine.mkdir(exist_ok=True)
        for name, tel in flagged:
            (raw_dir / name).rename(quarantine / name)
        print(f"[parallel] QA: quarantined {len(flagged)} rollouts "
              f"(> {args.max_teleports} teleports) → {quarantine}")
        for name, tel in flagged[:10]:
            print(f"           {name}: {tel} teleports")
    else:
        print(f"[parallel] QA: no rollouts exceeded {args.max_teleports} teleports.")

    # 4) Splits once.
    make_splits(
        raw_dir=str(raw_dir),
        splits_dir=str(splits_dir),
        config=ds_cfg["splits"],
        seed=int(ds.get("base_seed", ds.get("random_seed", 42))),
        stratify_by_family=True,
    )
    n_final = len(list(raw_dir.glob("*.npz")))
    print(f"[parallel] Done. {n_final} rollouts in {raw_dir}.")


if __name__ == "__main__":
    main()
