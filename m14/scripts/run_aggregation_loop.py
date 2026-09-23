"""
Train PPO in DeepONet and aggregate validation trajectories from SUMO.

Each round trains PPO, ranks checkpoints on surrogate validation return,
checks the top candidates in SUMO, appends their trajectories, and fine-tunes
the ensemble. A0 is the first round's best surrogate-selected checkpoint;
A1 is selected on SUMO validation return. Stop after --stop-patience rounds
with improvement below --stop-delta, or at --rounds.

Example: python scripts/run_aggregation_loop.py --study m14_s0 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.profile_eval import evaluate_on_profiles, load_set, print_summary, scenario_overlays, sumo_env_config  # noqa: E402
from rl.reward import RewardWeights  # noqa: E402
from sumo_env.rollout import load_rollout_npz  # noqa: E402
from sumo_env.rollout_store import RolloutStore  # noqa: E402
from surrogate.deeponet import DeepONetEnsemble  # noqa: E402
from surrogate.eval_plant import evaluate_rollouts  # noqa: E402
from train_ensemble import train_ensemble  # noqa: E402
from utils.config import load_config, merge_configs  # noqa: E402
from utils.ledger import Ledger  # noqa: E402

CATASTROPHIC = -150.0


def run_ppo(run_dir: Path, config: str, overlay: str, ensemble_dir: Path, steps: int, seed: int, init_policy: Path | None,
            extra_sets: list[str], log: Path) -> Path:
    cmd = [sys.executable, "-m", "rl.train_ppo", "--config", config, "--overlay", overlay]
    for ov in scenario_overlays():
        cmd += ["--overlay", ov]
    cmd += ["--ensemble-dir", str(ensemble_dir), "--seed", str(seed), "--total-timesteps", str(steps), "--set", f"output.run_dir={run_dir}"]
    if init_policy is not None:
        cmd += ["--init-policy", str(init_policy)]
    for s in extra_sets:
        cmd += ["--set", s]
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    with log.open("w") as f:
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"PPO run failed (code {rc}); see {log}")
    return run_dir


def rank_checkpoints(run_dir: Path, top_k: int) -> list[dict]:
    d = np.load(run_dir / "eval" / "evaluations.npz")
    cands = []
    for t, row in zip(d["timesteps"], d["results"]):
        ckpt = next(iter((run_dir / "checkpoints").glob(f"*_{int(t)}_steps.zip")), None)
        if ckpt is None:
            continue
        cands.append({"step": int(t), "path": str(ckpt), "surrogate_mean": float(row.mean()), "surrogate_min": float(row.min())})
    cands.sort(key=lambda c: -c["surrogate_mean"])
    return cands[:top_k], cands


def recover_interrupted_round(study_dir: Path, study: str, completed: int) -> Path:
    """Move an interrupted round's artifacts aside (nothing is deleted) so the loop can restart that round:
    round directories/files, the round's store entries and its ledger rows go to
    <study_dir>/interrupted_r<j>_<time>/. The moved SUMO episodes are not charged to the study, so the
    reported cost is that of an uninterrupted run; ledger_lost.jsonl keeps them for inspection."""
    aside = study_dir / f"interrupted_r{completed + 1}_{int(time.time())}"
    aside.mkdir()
    for path in list(study_dir.iterdir()):
        match = re.match(r"(?:ppo|rollouts|ensemble|agg|selected)_r(\d+)(?:[._]|$)", path.name)
        if match and int(match.group(1)) > completed:
            shutil.move(str(path), str(aside / path.name))
    if (study_dir / "store/index.json").exists():
        store = RolloutStore(study_dir / "store")
        late = [e for e in store.entries if int(e.get("round", 0)) > completed]
        if late:
            (aside / "store_entries.json").write_text(json.dumps(late, indent=1))
            store.entries = [e for e in store.entries if int(e.get("round", 0)) <= completed]
            store.save_index()
            store.write_split_index()
    ledger_path = PROJECT_ROOT / "runs/ledger" / f"{study}.jsonl"
    if ledger_path.exists():
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        late = [r for r in rows if int(r.get("round", 0)) > completed]
        if late:
            (aside / "ledger_lost.jsonl").write_text("".join(json.dumps(r) + "\n" for r in late))
            tmp = ledger_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows if int(r.get("round", 0)) <= completed))
            tmp.replace(ledger_path)
    print(f"[aggregation] recovered interrupted round {completed + 1}: artifacts moved to {aside}", flush=True)
    return aside


def completed_rounds_for_resume(study_dir: Path, study: str, resume: bool, recover: bool = False) -> list[dict]:
    """Guard: resume only at a completed round boundary (with `recover`, an interrupted round is moved aside)."""
    rounds_path = study_dir / "rounds.json"
    rows = json.loads(rounds_path.read_text()) if rounds_path.exists() else []
    guidance = ("No files were deleted. Keep this output for inspection and restart with a new --study name "
                "(and a new --out-dir if supplied). Resume supports completed round boundaries only.")
    if rows and not resume:
        raise RuntimeError(f"{study_dir} already contains recorded rounds; use --resume to continue. {guidance}")
    if [int(row["round"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"{rounds_path} does not contain contiguous completed rounds. {guidance}")
    for row in rows:
        checkpoint = Path(row["selected_path"])
        ensemble = Path(row["ensemble_after"])
        checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
        ensemble = ensemble if ensemble.is_absolute() else PROJECT_ROOT / ensemble
        if not checkpoint.is_file() or not (ensemble / "manifest.json").is_file():
            raise RuntimeError(f"Recorded round {row['round']} is missing its checkpoint or ensemble manifest. {guidance}")
    completed = len(rows)
    artifacts = []
    for path in study_dir.iterdir():
        match = re.match(r"(?:ppo|rollouts|ensemble|agg|selected)_r(\d+)(?:[._]|$)", path.name)
        if match and int(match.group(1)) > completed:
            artifacts.append(path.name)
    store_index = study_dir / "store/index.json"
    if completed and not store_index.is_file():
        raise RuntimeError(f"Completed rounds are missing their aggregation store index: {store_index}. {guidance}")
    if store_index.exists():
        entries = json.loads(store_index.read_text()).get("entries", [])
        if any(int(entry.get("round", 0)) > completed for entry in entries):
            artifacts.append("store/index.json has unrecorded aggregation trajectories")
    ledger_path = PROJECT_ROOT / "runs/ledger" / f"{study}.jsonl"
    if ledger_path.exists():
        ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        if any(int(row.get("round", 0)) > completed for row in ledger_rows):
            artifacts.append("study ledger has unrecorded aggregation episodes")
    if artifacts:
        if recover and resume:
            recover_interrupted_round(study_dir, study, completed)
            return rows
        raise RuntimeError(f"Incomplete round {completed + 1} artifacts under {study_dir}: "
                           + ", ".join(artifacts) + ". " + guidance
                           + " Or pass --recover-interrupted to move the incomplete round aside and redo it.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps-per-round", type=int, default=300_000)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--ensemble", default="runs/deeponet/round0")
    ap.add_argument("--store", default="data/round0")
    ap.add_argument("--surrogate-config", default="configs/deeponet.yaml")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--overlay", default="configs/env_surrogate.yaml")
    ap.add_argument("--set", action="append", default=[], help="extra --set overrides for the PPO runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stop-delta", type=float, default=2.0)
    ap.add_argument("--stop-patience", type=int, default=2,
                    help="stop after this many consecutive rounds with improvement < stop_delta (1 = original rule)")
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--val-set", default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-finetune", action="store_true", help="ablation: aggregation rollouts without model updates")
    ap.add_argument("--resume", action="store_true",
                    help="continue after complete recorded rounds; incomplete round artifacts cause an error and are preserved")
    ap.add_argument("--recover-interrupted", action="store_true",
                    help="with --resume: move an interrupted round's artifacts aside and redo the round (Colab sessions)")
    args = ap.parse_args()

    study_dir = PROJECT_ROOT / (args.out_dir or f"runs/aggregation/{args.study}")
    study_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_rounds_for_resume(study_dir, args.study, args.resume, args.recover_interrupted)
    if args.rounds < 1 or args.steps_per_round < 1 or args.top_k < 1 or args.stop_patience < 1:
        raise ValueError("rounds, steps-per-round, top-k and stop-patience must be positive")
    ledger = Ledger(args.study, PROJECT_ROOT)
    base_store = RolloutStore(PROJECT_ROOT / args.store)
    if args.resume and (study_dir / "store" / "index.json").exists():
        store = RolloutStore(study_dir / "store")            # keeps the aggregation rounds already appended
    else:
        store = base_store.fork(study_dir / "store")
    dataset_ee = sum(1 for e in base_store.entries if e["round"] == 0)
    profiles = load_set(args.val_set, PROJECT_ROOT)
    rcfg = merge_configs(load_config(str(PROJECT_ROOT / args.config)), load_config(str(PROJECT_ROOT / args.overlay)))
    for ov in scenario_overlays():
        rcfg = merge_configs(rcfg, load_config(str(PROJECT_ROOT / ov)))
    weights = RewardWeights.from_config(rcfg["env"]["reward"])
    warmup_s = float(rcfg["env"]["reward"].get("warmup_s", 90))

    ensemble_dir = PROJECT_ROOT / args.ensemble
    init_policy = None
    rounds_log = []
    best_prev = -np.inf
    bad_rounds = 0
    stopped_by_rule = False
    start_round = 1
    if completed:
        rounds_log = completed
        last = rounds_log[-1]
        ensemble_dir = Path(last["ensemble_after"])
        init_policy = Path(last["selected_path"])
        ensemble_dir = ensemble_dir if ensemble_dir.is_absolute() else PROJECT_ROOT / ensemble_dir
        init_policy = init_policy if init_policy.is_absolute() else PROJECT_ROOT / init_policy
        best_prev = max(float(row["best_sumo_val"]) for row in rounds_log)
        bad_rounds = int(last.get("bad_rounds", 0))
        stopped_by_rule = bad_rounds >= args.stop_patience
        start_round = int(last["round"]) + 1
        print(f"[resume] {len(rounds_log)} completed round(s) kept (best SUMO V {best_prev:.1f}, bad rounds {bad_rounds}); "
              f"next clean round is {start_round}, ensemble {ensemble_dir.name}, policy {init_policy.name}", flush=True)
    t_study = time.time()
    for j in range(start_round, args.rounds + 1):
        if stopped_by_rule:
            break
        t0 = time.time()
        print(f"\n===== {args.study}: round {j} / {args.rounds}  (ensemble {ensemble_dir.name}, store {len(store)} rollouts) =====", flush=True)
        # 1. PPO on the surrogate
        run_dir = study_dir / f"ppo_r{j}"
        run_ppo(run_dir, args.config, args.overlay, ensemble_dir, args.steps_per_round, args.seed + 100 * (j - 1), init_policy,
                args.set, study_dir / f"ppo_r{j}.log")
        t_ppo = time.time() - t0
        # 2. rank by surrogate V return
        top, all_cands = rank_checkpoints(run_dir, args.top_k)
        if not top:
            raise RuntimeError("no checkpoints found; is training.checkpoint_every_eval on?")
        print(f"[round {j}] PPO {t_ppo:.0f} s; top-{len(top)} by surrogate V return: " +
              ", ".join(f"{c['step']}:{c['surrogate_mean']:.1f}" for c in top), flush=True)
        # 3. SUMO rollouts of the top k on V
        env_cfg = sumo_env_config(PROJECT_ROOT, args.config, "configs/env_sumo.yaml",
                                  network_dir=str(study_dir / "network"), extra={"ensemble_dir": str(ensemble_dir),
                                  "observation": rcfg["env"].get("observation", {})})
        agg_dir = study_dir / f"rollouts_r{j}"
        res = evaluate_on_profiles([c["path"] for c in top], profiles, env_cfg, study_dir / f"agg_r{j}.jsonl", workers=args.workers,
                                   purpose="aggregation", study=args.study, round_index=j, save_rollouts_dir=agg_dir,
                                   project_root=PROJECT_ROOT, network_root=str(study_dir / "network"), quiet=True)
        summ = res["summary"]["policies"]
        print_summary(res["summary"], f"round {j}: SUMO validation of the top-{len(top)} checkpoints")
        for c in top:
            s = summ[c["path"]]
            c.update({"sumo_mean": s["mean"], "sumo_p10": s["p10"], "sumo_worst": s["worst"], "sumo_breakdown_rate": s["breakdown_rate"],
                      "n_catastrophic": s["n_catastrophic"], "gap": s["mean"] - c["surrogate_mean"]})
        feasible = [c for c in top if c["n_catastrophic"] == 0] or top
        selected = max(feasible, key=lambda c: c["sumo_mean"])
        sel_path = study_dir / f"selected_r{j}.zip"
        shutil.copy(selected["path"], sel_path)
        if j == 1:
            shutil.copy(top[0]["path"], study_dir / "A0_zero_shot.zip")   # best by surrogate return, no SUMO selection
        # 4. append + fine-tune
        new_files = []
        for r in res["results"]:
            meta = dict(r["meta"]); meta["round"] = j; meta["controller"] = {"type": "policy", "path": meta.get("policy"), "round": j}
            new_files.append((r["path"], meta))
        spread_before = evaluate_rollouts(DeepONetEnsemble.load(ensemble_dir), agg_dir, [Path(p).name for p, _ in new_files], weights, warmup_s)["summary"]["all"]
        store.append_round(new_files)
        if args.skip_finetune:
            new_ensemble = ensemble_dir
            spread_after = spread_before
        else:
            new_ensemble = study_dir / f"ensemble_r{j}"
            train_ensemble(args.surrogate_config, new_ensemble, resume_from=ensemble_dir, finetune=True, new_rounds=[j],
                           epochs=args.finetune_epochs, store_dir=str(store.root))
            spread_after = evaluate_rollouts(DeepONetEnsemble.load(new_ensemble), agg_dir, [Path(p).name for p, _ in new_files], weights, warmup_s)["summary"]["all"]
        # 5. log
        cum_ee = dataset_ee + ledger.budget()
        row = {"round": j, "ppo_run": str(run_dir), "ppo_wall_s": t_ppo, "candidates": all_cands, "top": top,
               "selected": selected, "selected_path": str(sel_path), "ensemble_before": str(ensemble_dir),
               "ensemble_after": str(new_ensemble), "spread_before": spread_before, "spread_after": spread_after,
               "dataset_ee": dataset_ee, "aggregation_ee": ledger.budget(), "cumulative_ee": cum_ee,
               "best_sumo_val": selected["sumo_mean"], "improvement": selected["sumo_mean"] - best_prev if np.isfinite(best_prev) else None,
               "stop_patience": args.stop_patience, "wall_s": time.time() - t0}
        rounds_log.append(row)
        (study_dir / "rounds.json").write_text(json.dumps(rounds_log, indent=1, default=str))
        print(f"[round {j}] selected {selected['step']} (surrogate {selected['surrogate_mean']:.1f} / SUMO {selected['sumo_mean']:.1f}, "
              f"gap {selected['gap']:+.1f}); spread on new rollouts {spread_before['return_rel_err_mean']:.3f} -> "
              f"{spread_after['return_rel_err_mean']:.3f} (return err); cumulative {cum_ee} EE; {row['wall_s']:.0f} s", flush=True)
        if np.isfinite(best_prev) and selected["sumo_mean"] - best_prev < args.stop_delta:
            bad_rounds += 1
            print(f"[round {j}] stop rule: improvement {selected['sumo_mean'] - best_prev:.1f} < {args.stop_delta} "
                  f"({bad_rounds}/{args.stop_patience})")
        else:
            bad_rounds = 0
        row["bad_rounds"] = bad_rounds
        (study_dir / "rounds.json").write_text(json.dumps(rounds_log, indent=1, default=str))
        best_prev = max(best_prev, selected["sumo_mean"])
        if bad_rounds >= args.stop_patience:
            stopped_by_rule = True
            break
        ensemble_dir = new_ensemble
        init_policy = sel_path
    final = max(rounds_log, key=lambda r: r["best_sumo_val"])
    shutil.copy(final["selected_path"], study_dir / "A1_final.zip")
    (study_dir / "study.json").write_text(json.dumps({"study": args.study, "seed": args.seed, "rounds": len(rounds_log),
                                                       "A0": str(study_dir / "A0_zero_shot.zip"), "A1": str(study_dir / "A1_final.zip"),
                                                       "A1_round": final["round"], "best_sumo_val": final["best_sumo_val"],
                                                       "cumulative_ee": final["cumulative_ee"], "wall_s": time.time() - t_study,
                                                       "stopped_by_rule": stopped_by_rule, "rounds_cap": args.rounds,
                                                       "stop_delta": args.stop_delta, "stop_patience": args.stop_patience}, indent=1))
    print(f"\n[{args.study}] done: A1 = round {final['round']} ({final['best_sumo_val']:.1f} on V at {final['cumulative_ee']} EE), "
          f"{time.time() - t_study:.0f} s total")


if __name__ == "__main__":
    main()
