"""
Launch M plant-model members as parallel processes and write the ensemble
manifest (M9, draft §5.4).

  PYTHONPATH=src python scripts/train_ensemble.py --config configs/surrogate/plant_v2.yaml \\
      --out-dir runs/surrogate/plant_v2_round0 [--members 5] [--epochs 150]

Fine-tune after aggregation round j (resume every member, add the new rounds):

  PYTHONPATH=src python scripts/train_ensemble.py --config configs/surrogate/plant_v2.yaml \\
      --out-dir runs/surrogate/plant_v2_round1 --resume-from runs/surrogate/plant_v2_round0 --finetune --new-rounds 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import load_config  # noqa: E402


def train_ensemble(config_path: str, out_dir: Path, members: int | None = None, resume_from: Path | None = None,
                   finetune: bool = False, new_rounds=None, epochs: int | None = None, store_dir: str | None = None,
                   max_train_files: int | None = None, parallel: int | None = None, extra_args=None) -> Path:
    cfg = load_config(str(PROJECT_ROOT / config_path))
    M = int(members or cfg.get("ensemble", {}).get("members", 5))
    if resume_from is not None:
        M = len(list(Path(resume_from).glob("member_*/best.pt"))) or M
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Another process may already be training this ensemble (member logs
    # updated within the last 15 min, no manifest yet): wait for it instead of
    # launching duplicate members into the same directory.
    logs = list(out_dir.glob("member_*.log"))
    if logs and not (out_dir / "manifest.json").exists():
        while any(time.time() - l.stat().st_mtime < 900 for l in logs) and not (out_dir / "manifest.json").exists():
            print(f"[ensemble] {out_dir.name}: training in progress elsewhere, waiting ...", flush=True)
            time.sleep(60)
        if (out_dir / "manifest.json").exists():
            print(f"[ensemble] {out_dir.name}: manifest appeared, reusing", flush=True)
            return out_dir
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    procs = []
    t0 = time.time()
    parallel = parallel or M
    pending = list(range(M))
    running = []
    while pending or running:
        while pending and len(running) < parallel:
            i = pending.pop(0)
            module = "surrogate.onestep" if str(cfg.get("model", {}).get("type", "")) == "onestep" or "onestep" in str(config_path) else "surrogate.train_plant"
            cmd = [sys.executable, "-m", module, "--config", config_path, "--member", str(i),
                   "--bootstrap-seed", str(i), "--out-dir", str(out_dir / f"member_{i}")]
            if resume_from is not None:
                cmd += ["--resume", str(Path(resume_from) / f"member_{i}" / "best.pt")]
            if finetune:
                cmd += ["--finetune"]
                if new_rounds:
                    cmd += ["--new-rounds", *[str(r) for r in new_rounds]]
            if epochs is not None:
                cmd += ["--epochs", str(epochs)]
            if store_dir:
                cmd += ["--store-dir", store_dir]
            if max_train_files is not None:
                cmd += ["--max-train-files", str(max_train_files)]
            cmd += list(extra_args or [])
            log = (out_dir / f"member_{i}.log").open("w")
            p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((i, p, log))
            print(f"[ensemble] member {i} started (pid {p.pid})", flush=True)
        time.sleep(2.0)
        still = []
        for i, p, log in running:
            if p.poll() is None:
                still.append((i, p, log))
            else:
                log.close()
                print(f"[ensemble] member {i} finished with code {p.returncode} ({time.time() - t0:.0f} s)", flush=True)
                if p.returncode != 0:
                    raise RuntimeError(f"member {i} failed; see {out_dir / f'member_{i}.log'}")
        running = still
    write_manifest(out_dir, cfg, config_path, store_dir, resume_from, new_rounds)
    print(f"[ensemble] manifest written: {out_dir / 'manifest.json'} ({time.time() - t0:.0f} s)")
    return out_dir


def write_manifest(out_dir: Path, cfg: dict, config_path: str, store_dir=None, resume_from=None, new_rounds=None) -> dict:
    import torch

    members = sorted(p for p in out_dir.glob("member_*/best.pt"))
    if not members:
        raise FileNotFoundError(f"no member checkpoints under {out_dir}")
    first = torch.load(str(members[0]), map_location="cpu", weights_only=False)
    bests = {}
    for p in members:
        bj = p.parent / "best.json"
        bests[p.parent.name] = json.loads(bj.read_text()) if bj.exists() else {}
    manifest = {
        "members": [str(p.relative_to(out_dir)) for p in members],
        "config_path": config_path,
        "store_dir": store_dir or cfg["data"]["store_dir"],
        "geometry": first["config"]["data"].get("geometry", {}),
        "normalization": first["normalization"],
        "model": first["config"]["model"],
        "resumed_from": None if resume_from is None else str(resume_from),
        "new_rounds": list(new_rounds or []),
        "val": bests,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/surrogate/plant_v2.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=None, help="max concurrent member processes")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--new-rounds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--max-train-files", type=int, default=None)
    args = ap.parse_args()
    train_ensemble(args.config, PROJECT_ROOT / args.out_dir, args.members,
                   None if args.resume_from is None else PROJECT_ROOT / args.resume_from, args.finetune,
                   args.new_rounds, args.epochs, args.store_dir, args.max_train_files, args.parallel)


if __name__ == "__main__":
    main()
