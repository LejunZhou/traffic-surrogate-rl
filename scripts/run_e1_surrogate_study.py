"""
E1 — surrogate study (draft §11): accuracy of the plant model versus
  (a) round-0 data size N_0 in {120, 240, 480(all)} training rollouts,
  (b) data source: random / open-loop only vs the behaviour mixture,
  (c) branch A (GRU, default) vs causal conv vs padded MLP (branch B),
  (d) ensemble size M = 1 vs 5 (evaluated from the M = 5 members),
  (e) DeepONet vs the one-step autoregressive model.
Every variant is trained on the same store and evaluated on the same
held-out split with scripts/eval_surrogate_regimes.py (return-prediction
error, breakdown metrics, calibration). Zero SUMO cost.

  PYTHONPATH=src python scripts/run_e1_surrogate_study.py --members 3 --epochs 150 [--only sizes,source,branch,onestep]
Writes _progress/m9_e1_surrogate_study.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.reward import RewardWeights  # noqa: E402
from surrogate.deeponet import DeepONetEnsemble  # noqa: E402
from surrogate.eval_plant import evaluate_rollouts, write_report  # noqa: E402
from train_ensemble import train_ensemble  # noqa: E402
from utils.config import load_config  # noqa: E402


def evaluate(ens_dir: Path, store: Path, files: list[str], weights, name: str, onestep: bool = False, members=None) -> dict:
    if onestep:
        from surrogate.onestep import OneStepEnsemble

        ens = OneStepEnsemble.load(ens_dir, members=members)
    else:
        ens = DeepONetEnsemble.load(ens_dir, members=members)
    res = evaluate_rollouts(ens, store, files, weights, warmup_s=90.0)
    write_report(res, ens_dir, name)
    return res["summary"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="data/plant_v2/round0")
    ap.add_argument("--base-ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default="sizes,source,branch,onestep")
    ap.add_argument("--out", default="_progress/m9_e1_surrogate_study.json")
    args = ap.parse_args()
    store = PROJECT_ROOT / args.store
    split = json.loads((store / "split_index.json").read_text())
    files = list(split[args.split])
    weights = RewardWeights.from_config(load_config(str(PROJECT_ROOT / "configs/rl/ppo_common.yaml"))["env"]["reward"])
    parts = set(args.only.split(","))
    out_path = PROJECT_ROOT / args.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def record(key, summary, extra=None):
        results[key] = {"summary": {k: v for k, v in summary.items() if k != "per_k_abs_err"}, **(extra or {})}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
        a, g = summary["all"], summary["gate"]
        print(f"[E1] {key:<28s} relL2 rho {a['rel_l2_density']:.3f} (band {a['rel_l2_band']:.3f}, jam {a['rel_l2_jam']:.3f}) "
              f"flow {a['rel_l2_flow']:.3f} return err {a['return_rel_err_mean']:.3f} false bd {a['false_breakdown_rate']:.3f} "
              f"calib {g['calibration_slope']:.2f} -> {'PASS' if g['passed'] else 'fail'}", flush=True)

    t0 = time.time()
    base = PROJECT_ROOT / args.base_ensemble
    if base.exists():
        record("gru_M5_all", evaluate(base, store, files, weights, f"e1_{args.split}"))
        record("gru_M1", evaluate(base, store, files, weights, f"e1_{args.split}_m1", members=[0]))
    if "sizes" in parts:
        for n in (120, 240):
            d = PROJECT_ROOT / f"runs/surrogate/e1_size{n}"
            if not (d / "manifest.json").exists():
                train_ensemble("configs/surrogate/plant_v2.yaml", d, args.members, epochs=args.epochs, max_train_files=n, parallel=args.parallel)
            record(f"gru_N{n}", evaluate(d, store, files, weights, f"e1_{args.split}"), {"n_train": n})
    if "source" in parts:
        d = PROJECT_ROOT / "runs/surrogate/e1_random_only"
        if not (d / "manifest.json").exists():
            train_ensemble("configs/surrogate/plant_v2.yaml", d, args.members, epochs=args.epochs, parallel=args.parallel,
                           extra_args=["--train-types", "constant", "random_signal"])
        record("gru_random_only", evaluate(d, store, files, weights, f"e1_{args.split}"))
    if "branch" in parts:
        for cfg, key in (("configs/surrogate/plant_v2_conv.yaml", "causal_conv"), ("configs/surrogate/plant_v2_mlp_padded.yaml", "mlp_padded")):
            d = PROJECT_ROOT / f"runs/surrogate/e1_{key}"
            if not (d / "manifest.json").exists():
                train_ensemble(cfg, d, args.members, epochs=args.epochs, parallel=args.parallel)
            record(key, evaluate(d, store, files, weights, f"e1_{args.split}"))
    if "onestep" in parts:
        d = PROJECT_ROOT / "runs/surrogate/onestep_round0"
        if not (d / "manifest.json").exists():
            train_ensemble("configs/surrogate/onestep_v1.yaml", d, args.members, parallel=args.parallel)
        record("onestep_mlp", evaluate(d, store, files, weights, f"e1_{args.split}", onestep=True))
    print(f"[E1] done in {time.time() - t0:.0f} s -> {out_path}")


if __name__ == "__main__":
    main()
