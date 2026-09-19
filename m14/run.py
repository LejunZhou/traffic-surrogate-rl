#!/usr/bin/env python3
"""M14: one entry point for simulation, data, learning, and evaluation.

Run ``python run.py --help``. Paths are relative to this folder, regardless
of the caller's working directory. Child processes use this copy of src/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
STUDY = "runs/study/m14"
ENSEMBLE = "runs/deeponet/round0"
STORE = "data/round0"


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PROFILE_SETS_DIR"] = str(ROOT / "configs/profiles")
    env.pop("SCENARIO_OVERLAY", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def execute(arguments: list[str], dry: bool = False, *, root: Path = ROOT) -> None:
    cmd = [sys.executable, *map(str, arguments)]
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry:
        return
    env = runtime_env()
    env["PYTHONPATH"] = str(root / "src")
    env["PROFILE_SETS_DIR"] = str(root / "configs/profiles")
    start = time.monotonic()
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "command": cmd,
              "cwd": str(root), "status": "failed"}
    try:
        subprocess.run(cmd, cwd=root, env=env, check=True)
        record["status"] = "completed"
    finally:
        record["elapsed_s"] = time.monotonic() - start
        (root / "runs").mkdir(exist_ok=True)
        with (root / "runs/commands.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def reused(path: str, dry: bool) -> bool:
    if not dry and (ROOT / path).is_file():
        print(f"Reusing {path}", flush=True)
        return True
    return False


def require(path: str, dry: bool, instruction: str) -> None:
    if not dry and not (ROOT / path).exists():
        raise RuntimeError(f"Missing {path}. {instruction}")


def check_environment() -> None:
    failures = []
    print(f"Project: {ROOT}\nPython:  {sys.version.split()[0]}")
    for module, dist in [("numpy", "numpy"), ("torch", "torch"),
                         ("stable_baselines3", "stable-baselines3"), ("gymnasium", "gymnasium"),
                         ("yaml", "PyYAML"), ("matplotlib", "matplotlib"), ("traci", "traci"), ("sumolib", "sumolib")]:
        try:
            importlib.import_module(module)
            print(f"  {dist}: {importlib.metadata.version(dist)}")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            failures.append(f"{dist}: {exc}")
    for binary in ("sumo", "netconvert"):
        path = shutil.which(binary, path=runtime_env()["PATH"])
        if path:
            version = subprocess.run([path, "--version"], capture_output=True, text=True,
                                     env=runtime_env(), check=True).stdout.splitlines()[0]
            print(f"  {version}")
        else:
            failures.append(f"{binary} is not on PATH")
    if not failures:
        for name in ("surrogate.deeponet", "rl.train_ppo", "sumo_env.rollout"):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(ROOT / "src"):
                failures.append(f"{name} imported from outside this folder: {module.__file__}")
    if failures:
        raise RuntimeError("\n".join(failures) + '\nInstall with: python -m pip install -e ".[sumo,dev]"')
    print("M14 runtime is ready. Source imports resolve inside this folder.")


def simulate(args) -> None:
    if args.dry_run:
        print(f"Run one {args.set} profile (index {args.index}) with {args.policy}; save {args.out}")
        return
    from rl.policy_specs import make_policy_callable
    from rl.profile_eval import load_set, sumo_env_config
    from rl.sumo_env_wrapper import SumoEnv
    from sumo_env.rollout import rollout_episode, save_rollout_npz
    # Constant/feedback controllers can be demonstrated before a dataset exists.
    # PPO evaluation must use the trained observation normalization.
    extra = {}
    is_feedback = args.policy.partition(":")[0] in ("alinea", "pialinea")
    if not (ROOT / STORE / "metadata.json").exists() and (is_feedback or args.policy.startswith("u=")):
        extra = {"density_mean": 20.0, "density_std": 12.0}
    cfg = sumo_env_config(ROOT, network_dir=str(ROOT / "data/networks/simulate"), extra=extra)
    profiles = load_set(args.set, ROOT)
    if not 0 <= args.index < len(profiles):
        raise ValueError(f"index must be between 0 and {len(profiles)-1}")
    profile = profiles[args.index]
    env = SumoEnv(cfg)
    try:
        controller = make_policy_callable(args.policy, env, cfg)
        result = rollout_episode(env, controller, {"profile": profile, "sumo_seed": args.seed})
        output = ROOT / args.out
        save_rollout_npz(output, result, {"profile": profile.to_dict(), "sumo_seed": args.seed,
                                       "controller": {"policy": args.policy}}, env)
        output.with_suffix(".json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2))
        print(f"Saved {output}")
    finally:
        env.close()


def data(args) -> None:
    if reused(f"{STORE}/split_index.json", args.dry_run):
        return
    if not reused(f"{STUDY}/e0.json", args.dry_run):
        execute(["scripts/run_scenario_characterisation.py", "--workers", args.workers,
                 "--study", "m14_e0", "--out", f"{STUDY}/e0.json"], args.dry_run)
    execute(["scripts/generate_round0_dataset.py", "--workers", args.workers,
             "--study", "m14_round0"], args.dry_run)


def deeponet(args) -> None:
    require(f"{STORE}/split_index.json", args.dry_run, "Run: python run.py data")
    if not reused(f"{ENSEMBLE}/manifest.json", args.dry_run):
        execute(["scripts/train_ensemble.py", "--out-dir", ENSEMBLE, "--members", args.members,
                 "--parallel", min(args.workers, args.members)], args.dry_run)
    for split in ("val", "test"):
        execute(["scripts/eval_surrogate_regimes.py", "--ensemble", ENSEMBLE,
                 "--store", STORE, "--split", split], args.dry_run)


def surrogate_ppo(args) -> None:
    require(f"{ENSEMBLE}/manifest.json", args.dry_run, "Run: python run.py deeponet")
    execute(["scripts/run_aggregation_loop.py", "--study", f"m14_s{args.seed}", "--seed", args.seed,
             "--rounds", args.rounds, "--steps-per-round", args.steps, "--workers", args.workers,
             "--stop-delta", "2", "--stop-patience", "2", "--resume"], args.dry_run)


def sumo_ppo(args) -> None:
    require(f"{STORE}/metadata.json", args.dry_run, "Run: python run.py data")
    for budget in args.budgets:
        run = f"{STUDY}/direct_ppo_{budget}ee_s{args.seed}"
        frequency = 4800 if budget <= 200 else (9600 if budget <= 700 else 24000)
        frequency = min(frequency, budget * 120)
        if not reused(f"{run}/final_model.zip", args.dry_run):
            if not args.dry_run and (ROOT / run).exists() and any((ROOT / run).iterdir()):
                raise RuntimeError(f"Incomplete direct PPO output at {run}. Use a fresh output directory via "
                                   "the low-level trainer or a fresh project copy; automatic restart would mix checkpoints and ledger entries.")
            if budget % 4:
                print("The requested budget is nominal: PPO collects full 480-step rollouts. "
                      "The ledger records the actual episode count.", flush=True)
            execute(["-m", "rl.train_ppo", "--config", "configs/ppo.yaml", "--overlay", "configs/env_sumo.yaml",
                     "--seed", args.seed, "--total-timesteps", budget * 120,
                     "--set", f"training.eval_freq={frequency}", "--set", f"output.run_dir={run}",
                     "--set", f"training.ledger_study=m14_direct_{budget}_s{args.seed}",
                     "--set", f"env.network_dir=data/networks/direct_{budget}_s{args.seed}"], args.dry_run)
        execute(["scripts/select_checkpoint_profiles.py", "--run", run], args.dry_run)


def baselines(args) -> None:
    require(f"{STORE}/metadata.json", args.dry_run, "Run: python run.py data")
    if not reused(f"{STUDY}/alinea_tuning.json", args.dry_run):
        execute(["scripts/tune_alinea_profiles.py", "--workers", args.workers, "--study", "m14_alinea",
                 "--out", f"{STUDY}/alinea_tuning.json"], args.dry_run)


def evaluate(args) -> None:
    require(f"{STUDY}/alinea_tuning.json", args.dry_run, "Run: python run.py baselines")
    tuning = json.loads((ROOT / f"{STUDY}/alinea_tuning.json").read_text()) if not args.dry_run else {
        "best_alinea": "<selected from alinea_tuning.json>", "best_constant": "<selected from alinea_tuning.json>"}
    execute(["scripts/build_arms_manifest.py", "--study", "m14", "--seeds", *args.seeds,
             "--direct-ee", *args.budgets, "--alinea", tuning["best_alinea"],
             "--constant", tuning["best_constant"], "--out", f"{STUDY}/arms.json"], args.dry_run)
    execute(["scripts/run_final_evaluation.py", "--arms", f"{STUDY}/arms.json", "--workers", args.workers,
             "--study", "m14_final", "--sets", *args.sets], args.dry_run)


def report(args) -> None:
    require(f"{STUDY}/arms.json", args.dry_run, "Run: python run.py evaluate")
    rounds = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "runs/aggregation").glob("m14_s*/rounds.json"))
    execute(["scripts/plot_sample_efficiency.py", "--arms", f"{STUDY}/arms.json",
             "--rounds", *rounds, "--out", "reports/figures"], args.dry_run)


def pipeline(args) -> None:
    args.seeds = [args.seed]
    for stage in (data, deeponet, surrogate_ppo, sumo_ppo, baselines, evaluate, report):
        stage(args)


def smoke(args) -> None:
    """Exercise the full path in a disposable copy, without touching study outputs."""
    if args.dry_run:
        print("Create an isolated copy in runs/smoke/<timestamp>; run 18 SUMO rollouts, train two DeepONets,\n"
              "one aggregation round, a short direct PPO run, and held-out evaluation. No full study is launched.")
        return
    import yaml
    scratch = ROOT / "runs/smoke" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    scratch.mkdir(parents=True)
    for directory in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / directory, scratch / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "run.py", scratch / "run.py")
    def modify(file, update):
        p = scratch / "configs" / file
        obj = yaml.safe_load(p.read_text()); update(obj)
        p.write_text(yaml.safe_dump(obj, sort_keys=False))
    def small_dataset(obj):
        obj["dataset"].update(n_rollouts=18, mixture={"constant": 1.0}, storage_mandatory_frac=0.0)
    modify("dataset.yaml", small_dataset)
    def short_training(obj):
        obj["training"].update(n_epochs=1, eval_every=1)
        obj["finetune"]["epochs"] = 1
    modify("deeponet.yaml", short_training)
    def small_ppo(obj):
        obj["ppo"].update(n_steps=120, batch_size=120, n_epochs=1)
        obj["training"].update(eval_freq=120)
    modify("ppo.yaml", small_ppo)
    modify("env_surrogate.yaml", lambda obj: (obj["env"].update(n_envs=2), obj["training"].update(eval_freq=120)))
    modify("env_sumo.yaml", lambda obj: obj["training"].update(eval_freq=120))
    for name in ("val", "test", "ood"):
        p = scratch / f"configs/profiles/{name}.json"
        obj = json.loads(p.read_text()); obj["profiles"] = obj["profiles"][:1]; obj["n"] = 1
        obj["profiles"][0]["sumo_seeds"] = [100]
        p.write_text(json.dumps(obj, indent=2))
    def run(*cmd): execute(list(cmd), root=scratch)
    run("scripts/generate_round0_dataset.py", "--workers", args.workers, "--study", "smoke_data")
    run("scripts/train_ensemble.py", "--out-dir", ENSEMBLE, "--members", 2, "--parallel", min(args.workers, 2))
    run("scripts/eval_surrogate_regimes.py", "--ensemble", ENSEMBLE, "--store", STORE, "--split", "test")
    run("scripts/run_aggregation_loop.py", "--study", "m14_s0", "--rounds", 1, "--steps-per-round", 240,
        "--top-k", 1, "--workers", args.workers, "--finetune-epochs", 1, "--set", "training.eval_freq=120")
    run("-m", "rl.train_ppo", "--config", "configs/ppo.yaml", "--overlay", "configs/env_sumo.yaml",
        "--total-timesteps", 240, "--set", f"output.run_dir={STUDY}/direct_ppo_2ee_s0",
        "--set", "training.ledger_study=m14_direct_2_s0")
    run("scripts/select_checkpoint_profiles.py", "--run", f"{STUDY}/direct_ppo_2ee_s0")
    run("scripts/tune_alinea_profiles.py", "--workers", args.workers, "--study", "m14_alinea", "--stage1-profiles", 0,
        "--dets", 13, "--rhos", 26, "--kis", 20, "--top", 1, "--out", f"{STUDY}/alinea_tuning.json")
    run("run.py", "evaluate", "--workers", args.workers, "--budgets", 2)
    run("run.py", "report")
    print(f"\nSmoke workflow passed. Artifacts: {scratch}\nThese short runs are integration checks, not scientific results.")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check dependencies, SUMO, and local imports")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="show the steps without running them")
    common.add_argument("--workers", type=positive_int, default=8, help="parallel SUMO workers (default: 8)")
    sim = sub.add_parser("simulate", parents=[common], help="save one SUMO trajectory before any training")
    sim.add_argument("--policy", default="u=0.5", help="u=0.5, an ALINEA spec, or a PPO .zip path")
    sim.add_argument("--set", choices=["val", "test", "ood"], default="val")
    sim.add_argument("--index", type=int, default=0)
    sim.add_argument("--seed", type=int, default=100)
    sim.add_argument("--out", default="runs/simulation/rollout.npz")
    descriptions = {"data": "collect E0 and mixture trajectories, then split the dataset",
                    "deeponet": "train the GRU ensemble and evaluate held-out trajectories",
                    "surrogate-ppo": "train PPO with SUMO validation and data aggregation",
                    "sumo-ppo": "train and select the direct SUMO-PPO baselines",
                    "baselines": "tune ALINEA/PI-ALINEA and constant metering on validation",
                    "evaluate": "evaluate available controllers on frozen ID/OOD profiles",
                    "report": "plot held-out returns against simulation cost",
                    "pipeline": "run the complete M14 study (hours of computation)",
                    "smoke": "test the end-to-end workflow in a separate temporary project"}
    for name, description in descriptions.items():
        p = sub.add_parser(name, help=description, parents=[common])
        if name in ("deeponet", "pipeline"):
            p.add_argument("--members", type=positive_int, default=5)
        if name in ("surrogate-ppo", "sumo-ppo", "pipeline"):
            p.add_argument("--seed", type=int, default=0)
        if name in ("surrogate-ppo", "pipeline"):
            p.add_argument("--rounds", type=positive_int, default=5)
            p.add_argument("--steps", type=positive_int, default=300000, help="surrogate PPO steps per round")
        if name in ("sumo-ppo", "evaluate", "pipeline"):
            p.add_argument("--budgets", type=positive_int, nargs="+", default=[200, 700], help="direct PPO training budgets in episodes")
        if name in ("evaluate", "pipeline"):
            if name == "evaluate":
                p.add_argument("--seeds", type=int, nargs="+", default=[0], help="trained policy seeds to include")
            p.add_argument("--sets", choices=["test", "ood", "val"], nargs="+", default=["test", "ood"])
    return ap


def main() -> None:
    args = parser().parse_args()
    os.chdir(ROOT)
    os.environ.pop("SCENARIO_OVERLAY", None)
    os.environ.update(runtime_env())
    commands = {"simulate": simulate, "data": data, "deeponet": deeponet, "surrogate-ppo": surrogate_ppo,
                "sumo-ppo": sumo_ppo, "baselines": baselines, "evaluate": evaluate, "report": report,
                "pipeline": pipeline, "smoke": smoke}
    try:
        if args.command == "check":
            check_environment()
        else:
            commands[args.command](args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"M14: {exc}") from exc


if __name__ == "__main__":
    main()
