"""
Budgeted data-aggregation loop (M10, draft_pipeline.md §6.2, steps 5-7).

Per round j = 1..R:
  1. PPO on the ensemble surrogate (S_j steps; round 1 from action_init_u,
     later rounds warm-started from the previous selected checkpoint)
  2. rank checkpoints by their surrogate return on V (free), take the top k
  3. roll the top k in SUMO on the 18 V profiles (k x 18 EE): the transfer
     measurement of the round AND the new on-policy data
  4. append the rollouts to the study's store (round j) and fine-tune every
     ensemble member (20 epochs, lr 3e-4)
  5. log surrogate vs SUMO return per checkpoint, ensemble spread on the new
     rollouts before / after fine-tuning, cumulative EE
Stop when the best SUMO validation return improves by < stop_delta, or after
R rounds. Arms: A0 = round-1 top checkpoint by surrogate return (zero-shot,
480 EE); A1(j) = round-j checkpoint with the best SUMO validation return.

  PYTHONPATH=src python scripts/run_aggregation_loop.py --study m10_s0 --seed 0 --rounds 4 \\
      --steps-per-round 1000000 --ensemble runs/surrogate/plant_v2_round0 --store data/plant_v2/round0
"""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps-per-round", type=int, default=1_000_000)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--ensemble", default="runs/surrogate/plant_v2_round0")
    ap.add_argument("--store", default="data/plant_v2/round0")
    ap.add_argument("--surrogate-config", default="configs/surrogate/plant_v2.yaml")
    ap.add_argument("--config", default="configs/rl/ppo_common.yaml")
    ap.add_argument("--overlay", default="configs/rl/env_surrogate.yaml")
    ap.add_argument("--set", action="append", default=[], help="extra --set overrides for the PPO runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stop-delta", type=float, default=2.0)
    ap.add_argument("--stop-patience", type=int, default=1,
                    help="stop after this many consecutive rounds with improvement < stop_delta (1 = original rule)")
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--val-set", default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-finetune", action="store_true", help="ablation: aggregation rollouts without model updates")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted study: keep the rounds recorded in rounds.json whose selected checkpoint and "
                         "fine-tuned ensemble exist, discard a half-finished next round, and carry on from there")
    args = ap.parse_args()

    study_dir = PROJECT_ROOT / (args.out_dir or f"runs/aggregation/{args.study}")
    study_dir.mkdir(parents=True, exist_ok=True)
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
    if args.resume and (study_dir / "rounds.json").exists():
        done = [r for r in json.loads((study_dir / "rounds.json").read_text())
                if Path(r["selected_path"]).exists() and (Path(r["ensemble_after"]) / "manifest.json").exists()]
        if done:
            rounds_log = done
            last = rounds_log[-1]
            ensemble_dir = Path(last["ensemble_after"])
            init_policy = Path(last["selected_path"])
            best_prev = max(float(r["best_sumo_val"]) for r in rounds_log)
            bad_rounds = int(last.get("bad_rounds", 0))
            stopped_by_rule = bad_rounds >= args.stop_patience
            start_round = int(last["round"]) + 1
            for stale in (study_dir / f"ppo_r{start_round}", study_dir / f"rollouts_r{start_round}", study_dir / f"ensemble_r{start_round}"):
                shutil.rmtree(stale, ignore_errors=True)
            (study_dir / f"ppo_r{start_round}.log").unlink(missing_ok=True)
            print(f"[resume] {len(rounds_log)} completed round(s) kept (best SUMO V {best_prev:.1f}, bad rounds {bad_rounds}); "
                  f"continuing from round {start_round} with ensemble {ensemble_dir.name} and policy {init_policy.name}", flush=True)
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
        env_cfg = sumo_env_config(PROJECT_ROOT, args.config, "configs/rl/env_sumo.yaml",
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
