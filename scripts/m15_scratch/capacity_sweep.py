"""M15 step 2: constant-demand capacity sweep on scenario v4 (3 lanes, 10 deg merge, LC2013
defaults), meter open (u = 1), 1-h episodes, 2 SUMO seeds per cell. Records the exit flow over
the last 30 min, the breakdown onset at the scenario threshold and at 25 / 35 veh/km on the
through-lane mean, and the max mean density. Output: _progress/m15_capacity_sweep_v4.json.
    python scripts/m15_scratch/capacity_sweep.py --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "scripts"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
from utils.config import load_config  # noqa: E402
from sumo_env.demand_profiles import DemandProfile  # noqa: E402
from sumo_env.parallel_rollouts import run_jobs  # noqa: E402
from sumo_env.rollout import breakdown_flags, load_rollout_npz  # noqa: E402

CELLS = [(d, 700) for d in (4800, 5000, 5100, 5200, 5300, 5400, 5600)] + \
        [(d, 400) for d in (5200, 5400, 5600, 5800)] + \
        [(d, 0) for d in (5800, 6200, 6600)] + \
        [(d, 800) for d in (4800, 5000, 5200)]
SEEDS = [7001, 7002]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "runs/m15/capacity_sweep"))
    ap.add_argument("--report", default=str(ROOT / "_progress/m15_capacity_sweep_v4.json"))
    args = ap.parse_args()
    env_cfg = dict(load_config(str(ROOT / "configs/experiments/round0_v3b.yaml"))["env"])
    env_cfg.update({"project_root": str(ROOT), "sumo_config": "configs/sumo/scenario_v4.yaml"})
    env_cfg.pop("density_stats_from", None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d, r in CELLS:
        for s in SEEDS:
            jobs.append({"name": f"cap_d{d}_r{r}_s{s}", "profile": DemandProfile.constant(float(d), float(r)).to_dict(),
                         "sumo_seed": s, "controller": {"type": "constant", "u": 1.0}, "round": 0, "purpose": "diagnostic",
                         "out_dir": str(out_dir)})
    jobs = [j for j in jobs if not (out_dir / f"{j['name']}.npz").exists()]
    print(f"[cap] {len(jobs)} rollouts on {args.workers} workers", flush=True)
    t0 = time.time()
    if jobs:
        run_jobs(jobs, env_cfg, workers=args.workers, network_root=str(out_dir / "network"))
    print(f"[cap] done in {time.time() - t0:.0f} s", flush=True)
    rows = []
    for d, r in CELLS:
        for s in SEEDS:
            arrays, meta = load_rollout_npz(out_dir / f"cap_d{d}_r{r}_s{s}.npz")
            m = meta["metrics"]
            rho = arrays["density"]
            K = rho.shape[1]
            row = {"mainline": d, "ramp": r, "offered": d + r, "seed": s,
                   "exit_vph_last30": float(np.mean(arrays["outflow_vph"][K // 2:])),
                   "served_frac": m["served_veh"] / max(m["offered_veh"], 1), "teleports": m["teleports"],
                   "rho_max_mean_lane": float(rho.max()), "rho_st11_last30": float(np.mean(rho[11, K // 2:])),
                   "rho_st12_last30": float(np.mean(rho[12, K // 2:])), "final_queue": m["final_queue"], "wall_s": m["wall_s"],
                   "bd_thr": m.get("breakdown_density_veh_km"), "breakdown": m["breakdown"], "onset_min": m["breakdown_onset_min"]}
            for thr in (25.0, 35.0):
                row[f"onset_min_thr{int(thr)}"] = breakdown_flags(rho, 30.0, threshold=thr)["breakdown_onset_min"]
            rows.append(row)
    Path(args.report).write_text(json.dumps({"cells": CELLS, "seeds": SEEDS, "rows": rows}, indent=1), encoding="utf-8")
    hdr = ("d", "r", "off", "seed", "exit30", "srv", "tel", "rhoMax", "st11", "st12", "bd", "on", "on25", "on35")
    print(" ".join(f"{h:>6}" for h in hdr))
    for w in rows:
        vals = (w["mainline"], w["ramp"], w["offered"], w["seed"], round(w["exit_vph_last30"]), round(w["served_frac"], 3),
                w["teleports"], round(w["rho_max_mean_lane"], 1), round(w["rho_st11_last30"], 1), round(w["rho_st12_last30"], 1),
                int(w["breakdown"]), w["onset_min"], w["onset_min_thr25"], w["onset_min_thr35"])
        print(" ".join(f"{v:>6}" for v in vals))
    print("report:", args.report)


if __name__ == "__main__":
    main()
