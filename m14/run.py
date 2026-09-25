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


def e0(args) -> None:
    """Scenario screening (its trajectories seed the initial dataset) and the capacity check
    against the 120 km/h-ramp reference report."""
    if not reused(f"{STUDY}/e0.json", args.dry_run):
        execute(["scripts/run_scenario_characterisation.py", "--workers", args.workers,
                 "--study", "m14_e0", "--out", f"{STUDY}/e0.json"], args.dry_run)
    execute(["scripts/compare_e0_capacity.py", "--new", f"{STUDY}/e0.json"], args.dry_run)


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
    # M14_RECOVER_INTERRUPTED=1 (set by `pipeline --recover-interrupted`, e.g. on Colab): a round cut off by a
    # lost session is moved aside and redone instead of stopping the study
    recover = ["--recover-interrupted"] if os.environ.get("M14_RECOVER_INTERRUPTED") == "1" else []
    execute(["scripts/run_aggregation_loop.py", "--study", f"m14_s{args.seed}", "--seed", args.seed,
             "--rounds", args.rounds, "--steps-per-round", args.steps, "--workers", args.workers,
             "--stop-delta", "2", "--stop-patience", "2", "--resume", *recover], args.dry_run)


def sumo_ppo(args) -> None:
    require(f"{STORE}/metadata.json", args.dry_run, "Run: python run.py data")
    for budget in args.budgets:
        run = f"{STUDY}/direct_ppo_{budget}ee_s{args.seed}"
        # validation every 40 episodes up to 200, every 80 above (the published 700/1000 runs used 9600 steps)
        frequency = 4800 if budget <= 200 else 9600
        frequency = min(frequency, budget * 120)
        if not reused(f"{run}/final_model.zip", args.dry_run):
            if not args.dry_run and (ROOT / run).exists() and any((ROOT / run).iterdir()):
                print(f"Resuming the interrupted direct PPO run at {run} from its latest checkpoint "
                      "(ledger and evaluation history are cut back to that checkpoint).", flush=True)
            if budget % 4:
                print("The requested budget is nominal: PPO collects full 480-step rollouts. "
                      "The ledger records the actual episode count.", flush=True)
            execute(["-m", "rl.train_ppo", "--config", "configs/ppo.yaml", "--overlay", "configs/env_sumo.yaml",
                     "--seed", args.seed, "--total-timesteps", budget * 120,
                     "--set", f"training.eval_freq={frequency}", "--set", f"output.run_dir={run}",
                     "--set", "training.resume=true",
                     "--set", f"training.ledger_study=m14_direct_{budget}_s{args.seed}",
                     "--set", f"env.network_dir=data/networks/direct_{budget}_s{args.seed}"], args.dry_run)
        execute(["scripts/select_checkpoint_profiles.py", "--run", run], args.dry_run)


def baselines(args) -> None:
    require(f"{STORE}/metadata.json", args.dry_run, "Run: python run.py data")
    if not reused(f"{STUDY}/alinea_tuning.json", args.dry_run):
        grid = []
        for flag in ("dets", "rhos", "kis", "kps"):
            values = getattr(args, flag, None)
            if values:
                grid += [f"--{flag}", *values]
        execute(["scripts/tune_alinea_profiles.py", "--workers", args.workers, "--study", "m14_alinea",
                 "--out", f"{STUDY}/alinea_tuning.json", *grid], args.dry_run)


def evaluate(args) -> None:
    require(f"{STUDY}/alinea_tuning.json", args.dry_run, "Run: python run.py baselines")
    tuning = json.loads((ROOT / f"{STUDY}/alinea_tuning.json").read_text()) if not args.dry_run else {
        "best_alinea": "<selected from alinea_tuning.json>", "best_constant": "<selected from alinea_tuning.json>",
        "best_pure_alinea": "<selected>", "best_pi_alinea": "<selected>"}
    extra = []
    if tuning.get("best_pure_alinea") and tuning.get("best_pi_alinea"):
        extra += ["--pure-alinea", tuning["best_pure_alinea"], "--pi-alinea", tuning["best_pi_alinea"]]
    if args.mpc:
        ensemble = final_ensemble(args.seeds[0], args.dry_run)
        extra += ["--mpc-spec", f"mpc:{ensemble},{args.mpc_args}" if args.mpc_args else f"mpc:{ensemble}"]
    execute(["scripts/build_arms_manifest.py", "--study", "m14", "--seeds", *args.seeds,
             "--direct-ee", *args.budgets, "--alinea", tuning["best_alinea"],
             "--constant", tuning["best_constant"], "--out", f"{STUDY}/arms.json", *extra], args.dry_run)
    execute(["scripts/run_final_evaluation.py", "--arms", f"{STUDY}/arms.json", "--workers", args.workers,
             "--study", "m14_final", "--sets", *args.sets], args.dry_run)


def final_ensemble(seed: int, dry: bool) -> str:
    """The last aggregation round's fine-tuned ensemble (Surrogate-MPC and Table I use it)."""
    rounds_path = ROOT / f"runs/aggregation/m14_s{seed}/rounds.json"
    if dry:
        return f"<last ensemble_after in {rounds_path.relative_to(ROOT)}>"
    if not rounds_path.exists():
        raise RuntimeError(f"Missing {rounds_path.relative_to(ROOT)}. Run: python run.py surrogate-ppo")
    ensemble = Path(json.loads(rounds_path.read_text())[-1]["ensemble_after"])
    return str(ensemble.relative_to(ROOT) if ensemble.is_relative_to(ROOT) else ensemble)


def tables(args) -> None:
    """Table I (final ensemble on the study store's val/test splits), Table II and the headline reductions.
    With several seeds: per-seed tables in tables/seed_<s>/ and their summary (mean ± sd, seed-level CIs)."""
    for seed in args.seeds:
        ensemble = final_ensemble(seed, args.dry_run)
        store = f"runs/aggregation/m14_s{seed}/store"
        for split in ("val", "test"):
            if args.dry_run or not (ROOT / ensemble / f"eval_{split}_rows.jsonl").exists():
                execute(["scripts/eval_surrogate_regimes.py", "--ensemble", ensemble, "--store", store, "--split", split],
                        args.dry_run)
    if len(args.seeds) > 1:
        execute(["scripts/build_paper_tables.py", "--arms", f"{STUDY}/arms.json", "--seeds", *args.seeds,
                 "--sets", *args.sets, "--out", f"{STUDY}/tables"], args.dry_run)
    else:
        execute(["scripts/build_paper_tables.py", "--arms", f"{STUDY}/arms.json", "--seed", args.seeds[0],
                 "--ensemble", final_ensemble(args.seeds[0], args.dry_run), "--sets", *args.sets,
                 "--out", f"{STUDY}/tables"], args.dry_run)


def report(args) -> None:
    require(f"{STUDY}/arms.json", args.dry_run, "Run: python run.py evaluate")
    rounds = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "runs/aggregation").glob("m14_s*/rounds.json"))
    execute(["scripts/plot_sample_efficiency.py", "--arms", f"{STUDY}/arms.json",
             "--rounds", *rounds, "--out", "reports/figures"], args.dry_run)


def _stage_argv(stage: str, args) -> list[str]:
    argv = ["--workers", str(args.workers)]
    if stage == "deeponet":
        argv += ["--members", str(args.members)]
    if stage in ("surrogate-ppo", "sumo-ppo"):
        argv += ["--seed", str(args.seed)]
    if stage == "surrogate-ppo":
        argv += ["--rounds", str(args.rounds), "--steps", str(args.steps)]
    if stage == "sumo-ppo":
        argv += ["--budgets", *map(str, args.budgets)]
    if stage == "baselines":
        for flag in ("dets", "rhos", "kis", "kps"):
            if getattr(args, flag, None):
                argv += [f"--{flag}", *map(str, getattr(args, flag))]
    return argv


def _run_branches(branches: dict[str, list[str]], args) -> None:
    """Run each branch's stages in order, the branches concurrently, as child run.py processes
    logging to runs/logs/pipeline_<branch>.log. Every stage reuses or resumes its outputs."""
    import threading

    (ROOT / "runs/logs").mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}

    def run_branch(name: str, stages: list) -> None:
        log_path = ROOT / f"runs/logs/pipeline_{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            for stage in stages:
                stage_args = args
                if isinstance(stage, tuple):              # (stage, seed): a branch for another policy seed
                    stage, seed = stage
                    stage_args = argparse.Namespace(**{**vars(args), "seed": seed})
                cmd = [sys.executable, str(ROOT / "run.py"), stage, *_stage_argv(stage, stage_args)]
                log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {shlex.join(cmd)}\n"); log.flush()
                code = subprocess.run(cmd, cwd=ROOT, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT).returncode
                if code != 0:
                    failures[name] = f"{stage} exited {code}; see {log_path.relative_to(ROOT)}"
                    return
        print(f"[pipeline] branch {name} finished", flush=True)

    threads = [threading.Thread(target=run_branch, args=item) for item in branches.items()]
    for name, stages in branches.items():
        steps = " -> ".join(f"{s[0]} --seed {s[1]}" if isinstance(s, tuple) else s for s in stages)
        print(f"[pipeline] branch {name}: {steps} (log runs/logs/pipeline_{name}.log)", flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise RuntimeError("parallel branches failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))


def pipeline(args) -> None:
    args.seeds = [args.seed]
    if args.recover_interrupted:
        os.environ["M14_RECOVER_INTERRUPTED"] = "1"      # inherited by every stage and child process
    e0(args)
    if not args.dry_run and not args.ignore_capacity_check:
        check = json.loads((ROOT / f"{STUDY}/e0_capacity_comparison.json").read_text())
        if check["rescale_recommended"] and not (ROOT / f"{STORE}/split_index.json").exists():
            raise RuntimeError(f"capacity check: {check['verdict']} (mean shift {check['mean_shift_vph']:+.0f} veh/h). "
                               "Rescale the constants, or rerun with --ignore-capacity-check to proceed anyway.")
    data(args)
    branches = {"surrogate": ["deeponet", "surrogate-ppo"], "direct": ["sumo-ppo"], "baselines": ["baselines"]}
    if args.parallel and not args.dry_run:
        _run_branches(branches, args)
    else:
        for stage in (deeponet, surrogate_ppo, sumo_ppo, baselines):
            stage(args)
    for stage in (evaluate, tables, report):
        stage(args)


def seeds(args) -> None:
    """More policy seeds for a finished seed-0 study. The initial data, the round-0 ensemble and the tuned
    baselines are shared; each new seed's surrogate-PPO loop and direct SUMO-PPO run as concurrent branches
    (logs runs/logs/pipeline_{surrogate,direct}_s<seed>.log), then seed 0 and the new seeds are evaluated
    (seed 0's results are reused) and tabulated. Use --torch-threads so the branches share the CPU."""
    for path, instruction in ((f"{STORE}/split_index.json", "Run: python run.py data"),
                              (f"{ENSEMBLE}/manifest.json", "Run: python run.py deeponet"),
                              (f"{STUDY}/alinea_tuning.json", "Run: python run.py baselines"),
                              ("runs/aggregation/m14_s0/rounds.json", "Run the seed-0 study first (python run.py pipeline)")):
        require(path, args.dry_run, instruction)
    if args.recover_interrupted:
        os.environ["M14_RECOVER_INTERRUPTED"] = "1"      # inherited by every stage and child process
    new = [s for s in dict.fromkeys(args.new_seeds) if s != 0]
    branches = {}
    for s in new:
        branches[f"surrogate_s{s}"] = [("surrogate-ppo", s)]
        branches[f"direct_s{s}"] = [("sumo-ppo", s)]
    if args.dry_run:
        for s in new:
            for stage in (surrogate_ppo, sumo_ppo):
                stage(argparse.Namespace(**{**vars(args), "seed": s}))
    else:
        _run_branches(branches, args)
    args.seeds = [0, *new]
    for stage in (evaluate, tables, report):
        stage(args)


def _continue_direct_run(seed: int, source_budget: int, budget: int, dry: bool) -> None:
    """Start direct_ppo_<budget>ee_s<seed> as a copy of the finished <source_budget> run: its final model
    (policy, value net, optimizer, step count) becomes the checkpoint `sumo-ppo` resumes from, and the
    evaluation history and SUMO ledger come along, so the longer run is charged for all of its episodes
    and its checkpoint selection sees every evaluation since step 0."""
    import zipfile

    source, target = ROOT / f"{STUDY}/direct_ppo_{source_budget}ee_s{seed}", ROOT / f"{STUDY}/direct_ppo_{budget}ee_s{seed}"
    ledger_dir = ROOT / "runs/ledger"
    source_ledger = ledger_dir / f"m14_direct_{source_budget}_s{seed}.jsonl"
    target_ledger = ledger_dir / f"m14_direct_{budget}_s{seed}.jsonl"
    if dry:
        print(f"$ copy {source.relative_to(ROOT)} -> {target.relative_to(ROOT)} (final_model.zip as its latest "
              f"checkpoint) and {source_ledger.relative_to(ROOT)} -> {target_ledger.name}", flush=True)
        return
    if target.exists():
        print(f"{target.relative_to(ROOT)} exists: continuing it as it is", flush=True)
        return
    if not (source / "final_model.zip").exists():
        raise RuntimeError(f"{source.relative_to(ROOT)}/final_model.zip missing: finish the "
                           f"{source_budget}-episode run first (python run.py sumo-ppo --seed {seed} --budgets {source_budget})")
    with zipfile.ZipFile(source / "final_model.zip") as z:
        steps = int(json.loads(z.read("data"))["num_timesteps"])
    staging = target.with_name(target.name + ".partial")      # a copy cut off by a lost session is redone
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("final_model.zip", "best_model_selected.zip",
                                                                   "selection.json"))
    shutil.copyfile(source / "final_model.zip", staging / "checkpoints" / f"ppo_sumo_{steps}_steps.zip")
    rows = [json.loads(line) for line in source_ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    target_ledger.write_text("".join(json.dumps({**r, "study": f"m14_direct_{budget}_s{seed}"}) + "\n" for r in rows),
                             encoding="utf-8")
    (staging / "continued_from.json").write_text(json.dumps(
        {"source": str(source.relative_to(ROOT)), "steps": steps, "ledger_rows": len(rows),
         "time": datetime.now(timezone.utc).isoformat()}, indent=1))
    staging.rename(target)
    print(f"{target.relative_to(ROOT)}: continues {source.name} from step {steps} ({len(rows)} ledger rows)", flush=True)


def extend_direct(args) -> None:
    """Continue the finished direct SUMO-PPO runs of every seed to a larger episode budget (same seed, same
    optimizer state, constant learning rate: equivalent to having trained longer), then evaluate and tabulate
    both budgets. Table II's SUMO-PPO row becomes the larger budget; the smaller one gets its own row."""
    require(f"{STUDY}/alinea_tuning.json", args.dry_run, "Run: python run.py baselines")
    if args.to_budget <= args.from_budget:
        raise ValueError("--to-budget must be larger than --from-budget")
    if args.recover_interrupted:
        os.environ["M14_RECOVER_INTERRUPTED"] = "1"      # inherited by every stage and child process
    args.seeds = list(dict.fromkeys(args.seeds))
    for s in args.seeds:
        _continue_direct_run(s, args.from_budget, args.to_budget, args.dry_run)
    args.budgets = [args.to_budget]
    if args.dry_run:
        for s in args.seeds:
            sumo_ppo(argparse.Namespace(**{**vars(args), "seed": s}))
    else:
        _run_branches({f"direct_s{s}": [("sumo-ppo", s)] for s in args.seeds}, args)
    args.budgets = [args.from_budget, args.to_budget]
    for stage in (evaluate, tables, report):
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
    direct = [
        "-m", "rl.train_ppo", "--config", "configs/ppo.yaml", "--overlay", "configs/env_sumo.yaml",
        "--set", f"output.run_dir={STUDY}/direct_ppo_2ee_s0", "--set", "training.ledger_study=m14_direct_2_s0",
        "--set", "training.resume=true"]
    run(*direct, "--total-timesteps", 240)
    # simulate an interrupted session: the run lost its final model and continues to 480 steps from its checkpoint
    (scratch / STUDY / "direct_ppo_2ee_s0/final_model.zip").unlink()
    run(*direct, "--total-timesteps", 480)
    _check_resumed_direct_run(scratch / STUDY / "direct_ppo_2ee_s0", scratch / "runs/ledger/m14_direct_2_s0.jsonl")
    run("scripts/select_checkpoint_profiles.py", "--run", f"{STUDY}/direct_ppo_2ee_s0")
    run("scripts/tune_alinea_profiles.py", "--workers", args.workers, "--study", "m14_alinea", "--stage1-profiles", 0,
        "--dets", 13, "--rhos", 26, "--kis", 20, "--top", 1, "--out", f"{STUDY}/alinea_tuning.json")
    run("run.py", "evaluate", "--workers", args.workers, "--budgets", 2, "--mpc-args", "H=4,iters=2")
    run("run.py", "tables")
    run("run.py", "report")
    print(f"\nSmoke workflow passed. Artifacts: {scratch}\nThese short runs are integration checks, not scientific results.")


def _check_resumed_direct_run(run: Path, ledger: Path) -> None:
    """Smoke assertions: 4 checkpoints and 4 evaluations at 120-step spacing, ledger = 4 training
    + 4 validation episodes (the smoke validation set has one profile)."""
    import numpy as np

    steps = sorted(int(p.stem.split("_")[-2]) for p in (run / "checkpoints").glob("*_steps.zip"))
    evals = np.load(run / "eval/evaluations.npz")["timesteps"].tolist()
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    purposes = {p: sum(r["purpose"] == p for r in rows) for p in ("direct_ppo", "eval_val")}
    problems = []
    if steps != [120, 240, 360, 480]:
        problems.append(f"checkpoints {steps}")
    if evals != [120, 240, 360, 480]:
        problems.append(f"evaluations {evals}")
    if purposes != {"direct_ppo": 4, "eval_val": 4}:
        problems.append(f"ledger {purposes}")
    if not (run / "final_model.zip").exists() or not (run / "resume_log.jsonl").exists():
        problems.append("final_model.zip or resume_log.jsonl missing")
    if problems:
        raise RuntimeError("resumed direct PPO run is inconsistent: " + "; ".join(problems))
    print(f"Resume check passed: checkpoints {steps}, evaluations {evals}, ledger {purposes}", flush=True)


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
    common.add_argument("--torch-threads", type=positive_int, default=None,
                        help="CPU threads per PPO training process (sets M14_TORCH_THREADS for every child process; "
                             "default: all cores); use it when several studies share one machine")
    sim = sub.add_parser("simulate", parents=[common], help="save one SUMO trajectory before any training")
    sim.add_argument("--policy", default="u=0.5", help="u=0.5, an ALINEA spec, or a PPO .zip path")
    sim.add_argument("--set", choices=["val", "test", "ood"], default="val")
    sim.add_argument("--index", type=int, default=0)
    sim.add_argument("--seed", type=int, default=100)
    sim.add_argument("--out", default="runs/simulation/rollout.npz")
    descriptions = {"e0": "scenario screening and the capacity check against the 120 km/h-ramp reference",
                    "data": "collect E0 and mixture trajectories, then split the dataset",
                    "deeponet": "train the GRU ensemble and evaluate held-out trajectories",
                    "surrogate-ppo": "train PPO with SUMO validation and data aggregation",
                    "sumo-ppo": "train and select the direct SUMO-PPO baselines",
                    "baselines": "tune ALINEA/PI-ALINEA and constant metering on validation",
                    "evaluate": "evaluate available controllers on frozen ID/OOD profiles",
                    "tables": "build the paper's Table I, Table II and headline TTS reductions",
                    "report": "plot held-out returns against simulation cost",
                    "pipeline": "run the complete M14 study (hours of computation)",
                    "seeds": "train more policy seeds on the finished seed-0 study in parallel, then evaluate and tabulate all seeds",
                    "extend-direct": "continue the finished direct SUMO-PPO runs to a larger episode budget, then evaluate and tabulate",
                    "smoke": "test the end-to-end workflow in a separate temporary project"}
    for name, description in descriptions.items():
        p = sub.add_parser(name, help=description, parents=[common])
        if name in ("deeponet", "pipeline"):
            p.add_argument("--members", type=positive_int, default=5)
        if name in ("surrogate-ppo", "sumo-ppo", "pipeline"):
            p.add_argument("--seed", type=int, default=0)
        if name == "seeds":
            p.add_argument("--new-seeds", type=int, nargs="+", default=[1, 2], help="policy seeds to add (default: 1 2)")
        if name in ("surrogate-ppo", "pipeline", "seeds"):
            p.add_argument("--rounds", type=positive_int, default=5)
            p.add_argument("--steps", type=positive_int, default=300000, help="surrogate PPO steps per round")
        if name in ("sumo-ppo", "evaluate", "pipeline", "seeds"):
            p.add_argument("--budgets", type=positive_int, nargs="+", default=[1000], help="direct PPO training budgets in episodes")
        if name == "extend-direct":
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="policy seeds to continue (default: 0 1 2)")
            p.add_argument("--from-budget", type=positive_int, default=1000, help="finished direct PPO budget, episodes")
            p.add_argument("--to-budget", type=positive_int, default=1200, help="budget to continue to, episodes")
        if name in ("evaluate", "tables", "pipeline", "seeds", "extend-direct"):
            if name in ("evaluate", "tables"):
                p.add_argument("--seeds", type=int, nargs="+", default=[0], help="trained policy seeds to include")
            p.add_argument("--sets", choices=["test", "ood", "val"], nargs="+", default=["test", "ood"])
        if name in ("evaluate", "pipeline", "seeds", "extend-direct"):
            p.add_argument("--mpc", action=argparse.BooleanOptionalAction, default=True,
                           help="evaluate Surrogate-MPC on the last aggregation ensemble (default: on)")
            p.add_argument("--mpc-args", default="iters=30", help="MPC options appended to the spec, e.g. H=20,iters=30")
        if name in ("baselines", "pipeline"):
            p.add_argument("--dets", type=int, nargs="+", default=None, help="ALINEA detector stations (default: tuner's)")
            p.add_argument("--rhos", type=float, nargs="+", default=None, help="ALINEA target densities, veh/km")
            p.add_argument("--kis", type=float, nargs="+", default=None, help="ALINEA integral gains")
            p.add_argument("--kps", type=float, nargs="+", default=None, help="PI-ALINEA proportional gains")
        if name in ("seeds", "extend-direct"):
            p.add_argument("--recover-interrupted", action="store_true",
                           help="rerun work cut off by a lost session (see pipeline --recover-interrupted)")
        if name == "pipeline":
            p.add_argument("--parallel", action="store_true",
                           help="after the data stage run [deeponet -> surrogate-ppo], sumo-ppo and baselines concurrently")
            p.add_argument("--ignore-capacity-check", action="store_true",
                           help="generate data even if the E0 capacity check recommends rescaling")
            p.add_argument("--recover-interrupted", action="store_true",
                           help="rerun work cut off by a lost session (interrupted aggregation rounds and evaluation "
                                "requests are moved aside, never deleted); use when no other study process is running")
    return ap


def main() -> None:
    args = parser().parse_args()
    os.chdir(ROOT)
    os.environ.pop("SCENARIO_OVERLAY", None)
    if getattr(args, "torch_threads", None):
        os.environ["M14_TORCH_THREADS"] = str(args.torch_threads)     # read by rl.train_ppo in every child process
    os.environ.update(runtime_env())
    commands = {"simulate": simulate, "e0": e0, "data": data, "deeponet": deeponet, "surrogate-ppo": surrogate_ppo,
                "sumo-ppo": sumo_ppo, "baselines": baselines, "evaluate": evaluate, "tables": tables, "report": report,
                "pipeline": pipeline, "seeds": seeds, "extend-direct": extend_direct, "smoke": smoke}
    try:
        if args.command == "check":
            check_environment()
        else:
            commands[args.command](args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"M14: {exc}") from exc


if __name__ == "__main__":
    main()
