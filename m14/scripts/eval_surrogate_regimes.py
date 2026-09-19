"""
Evaluate DeepONet density, exit-flow, return, and regime errors.

Example: python scripts/eval_surrogate_regimes.py
         --ensemble runs/deeponet/round0 --store data/round0 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl.reward import RewardWeights  # noqa: E402
from surrogate.deeponet import DeepONetEnsemble  # noqa: E402
from surrogate.eval_plant import evaluate_rollouts, print_summary, write_report  # noqa: E402
from utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--store", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--rounds", type=int, nargs="*", default=None, help="restrict to these store rounds")
    ap.add_argument("--reward-config", default="configs/ppo.yaml")
    ap.add_argument("--members", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    ens = DeepONetEnsemble.load(PROJECT_ROOT / args.ensemble, members=args.members)
    store = Path(args.store or ens.manifest.get("store_dir", "data/round0"))
    store = store if store.is_absolute() else PROJECT_ROOT / store
    if args.files:
        files = args.files
    else:
        split = json.loads((store / "split_index.json").read_text())
        files = list(split[args.split])
        if args.rounds is not None:
            index = {e["file"]: e for e in json.loads((store / "index.json").read_text())["entries"]}
            files = [f for f in files if index[f]["round"] in set(args.rounds)]
    rcfg = load_config(str(PROJECT_ROOT / args.reward_config))
    reward_cfg = rcfg.get("env", rcfg).get("reward", {})
    weights = RewardWeights.from_config(reward_cfg)
    res = evaluate_rollouts(ens, store, files, weights, warmup_s=float(reward_cfg.get("warmup_s", 90)))
    name = args.name or f"eval_{args.split}"
    out = write_report(res, PROJECT_ROOT / (args.out or args.ensemble), name)
    print_summary(res["summary"], f"{args.ensemble} on {args.split} ({len(files)} rollouts)")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
