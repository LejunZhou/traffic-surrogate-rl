"""
Train independent bootstrap DeepONet members and write their manifest.

Example: python scripts/train_ensemble.py --config configs/deeponet.yaml
         --out-dir runs/deeponet/round0 --members 5 --parallel 5
Fine-tuning uses --resume-from, --finetune and --new-rounds.
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
    # A complete manifest is the stage completion marker. Failed members are
    # retried immediately; only members with an explicit success marker are reused.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all((out_dir / member).exists() for member in manifest["members"]):
            print(f"[ensemble] reusing complete ensemble: {out_dir}", flush=True)
            return out_dir
        raise RuntimeError(f"incomplete manifest under {out_dir}; remove or repair it before retrying")
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    t0 = time.time()
    parallel = parallel or M
    if M < 1 or parallel < 1:
        raise ValueError("members and parallel must be positive")
    pending = [i for i in range(M) if not ((out_dir / f"member_{i}" / "completed.json").exists()
                                         and (out_dir / f"member_{i}" / "best.pt").exists())]
    running = []
    try:
        while pending or running:
            while pending and len(running) < parallel:
                i = pending.pop(0)
                cmd = [sys.executable, "-m", "surrogate.train_plant", "--config", config_path,
                       "--member", str(i), "--bootstrap-seed", str(i),
                       "--out-dir", str(out_dir / f"member_{i}")]
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
                try:
                    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                running.append((i, proc, log))
                print(f"[ensemble] member {i} started (pid {proc.pid})", flush=True)
            time.sleep(0.2)
            completed = []
            for i, proc, log in running:
                if proc.poll() is None:
                    continue
                log.close()
                if proc.returncode != 0:
                    raise RuntimeError(f"member {i} failed; see {out_dir / f'member_{i}.log'}; "
                                       "rerun the same command to retry unfinished members")
                member_dir = out_dir / f"member_{i}"
                if not (member_dir / "best.pt").exists():
                    raise RuntimeError(f"member {i} exited without best.pt; see {out_dir / f'member_{i}.log'}")
                (member_dir / "completed.json").write_text(json.dumps({"returncode": 0, "completed_at": time.time()}))
                completed.append(i)
                print(f"[ensemble] member {i} finished ({time.time() - t0:.0f} s)", flush=True)
            running = [item for item in running if item[0] not in completed]
    except BaseException:
        for _, proc, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, log in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        raise
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
    ap.add_argument("--config", default="configs/deeponet.yaml")
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
