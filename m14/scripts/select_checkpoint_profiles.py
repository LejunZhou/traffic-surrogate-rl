"""
Select a direct-SUMO PPO checkpoint on validation return.

Example: python scripts/select_checkpoint_profiles.py
         --run runs/study/m14/direct_ppo_700ee_s0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--catastrophic", type=float, default=-250.0)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else PROJECT_ROOT / args.run
    npz = run / "eval" / "evaluations.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz)
    cands = []
    for t, row in zip(d["timesteps"], d["results"]):
        ckpt = next(iter((run / "checkpoints").glob(f"*_{int(t)}_steps.zip")), None)
        if ckpt is None:
            continue
        cands.append({"step": int(t), "path": str(ckpt), "mean": float(row.mean()), "min": float(row.min()),
                      "catastrophic": int((row < args.catastrophic).sum())})
    if not cands:
        raise RuntimeError("no checkpoints matched the evaluation steps")
    feasible = [c for c in cands if c["catastrophic"] == 0] or cands
    best = max(feasible, key=lambda c: c["mean"])
    shutil.copy(best["path"], run / "best_model_selected.zip")
    (run / "eval" / "selection.json").write_text(json.dumps({"chosen": best, "candidates": cands}, indent=1))
    print(f"{run.name}: selected step {best['step']} (V mean {best['mean']:.1f}, min {best['min']:.1f}) -> best_model_selected.zip")


if __name__ == "__main__":
    main()
