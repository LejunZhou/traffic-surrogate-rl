"""
Milestone 5b sweep driver.

For each (beta, seed) in the cartesian product:
1. Run `python -m rl.train_ppo` as a subprocess with the override flags
   --reward-beta, --seed, --run-name-suffix (and optional --total-timesteps).
2. Find the resulting run dir under runs/ppo/.
3. Use scripts/eval_constant_baselines.py's rollout_policy to evaluate:
   - The learned best_model.zip at the training beta
   - Constant baselines u=0.0, u=0.5, u=1.0 at the same beta
4. Append a row to results.csv.

After all combinations, prints a per-beta summary table:
  - mean / std of learned-policy reward across seeds
  - win rate over each constant baseline
  - applies the acceptance criterion (>=4/5 wins over u=1.0 + 5% margin)

Usage:
  python scripts/run_m5b_sweep.py --betas 0.3,1.0,3.0 --seeds 0,1,2,3,4
  python scripts/run_m5b_sweep.py --betas 0.5 --seeds 0 --total-timesteps 2000  # smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_constant_baselines import rollout_policy  # type: ignore  # noqa: E402


CSV_FIELDS = [
    "beta",
    "seed",
    "run_dir",
    "learned_reward",
    "learned_action_mean",
    "learned_action_std",
    "learned_density_mean",
    "learned_queue_final",
    "u0_reward",
    "u05_reward",
    "u1_reward",
    "beats_u0",
    "beats_u05",
    "beats_u1",
    "best_baseline_reward",
    "margin_over_best_baseline",
]


def _parse_csv_arg(arg: str, cast):
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


def _suffix(beta: float, seed: int) -> str:
    return f"b{beta:.2f}_s{seed}".replace(".", "p")


def _find_run_dir(suffix: str, after: dt.datetime) -> Path:
    """Locate the run dir whose name contains `suffix` and is newer than `after`."""
    runs_root = PROJECT_ROOT / "runs" / "ppo"
    matches = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        if suffix not in path.name:
            continue
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
        if mtime >= after:
            matches.append((mtime, path))
    if not matches:
        raise FileNotFoundError(
            f"No run dir matching suffix='{suffix}' newer than {after.isoformat()}"
        )
    matches.sort()
    return matches[-1][1]


def _train_one(
    beta: float,
    seed: int,
    config_path: str,
    total_timesteps: int | None,
    log_dir: Path,
) -> Path:
    """Spawn one PPO training run with the requested overrides."""
    suffix = _suffix(beta, seed)
    cmd = [
        sys.executable,
        "-m",
        "rl.train_ppo",
        "--config",
        config_path,
        "--reward-beta",
        str(beta),
        "--seed",
        str(seed),
        "--run-name-suffix",
        suffix,
    ]
    if total_timesteps is not None:
        cmd.extend(["--total-timesteps", str(total_timesteps)])

    log_file = log_dir / f"train_{suffix}.log"
    start = dt.datetime.now()
    print(f"[sweep] beta={beta} seed={seed} → spawning, log: {log_file}")
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"train_ppo failed for beta={beta} seed={seed} (exit={proc.returncode}). "
            f"See {log_file}."
        )
    run_dir = _find_run_dir(suffix, start)
    print(f"[sweep] beta={beta} seed={seed} → run_dir={run_dir.name}")
    return run_dir


def _evaluate_run(run_dir: Path, beta: float, seed: int) -> dict:
    """Roll out the learned policy and three constant baselines at the same beta."""
    best_zip = run_dir / "best_model.zip"
    if not best_zip.exists():
        # Fall back to final_model if best wasn't saved (no EvalCallback hits)
        alt = run_dir / "final_model.zip"
        if alt.exists():
            best_zip = alt
        else:
            raise FileNotFoundError(
                f"No best_model.zip or final_model.zip in {run_dir}"
            )

    learned = rollout_policy(
        policy_arg=str(best_zip), seed=seed, beta=beta, alpha=1.0, gamma=1.0
    )
    u0 = rollout_policy("u=0.0", seed=seed, beta=beta, alpha=1.0, gamma=1.0)
    u05 = rollout_policy("u=0.5", seed=seed, beta=beta, alpha=1.0, gamma=1.0)
    u1 = rollout_policy("u=1.0", seed=seed, beta=beta, alpha=1.0, gamma=1.0)

    best_baseline = max(
        u0["total_reward"], u05["total_reward"], u1["total_reward"]
    )
    return {
        "beta": beta,
        "seed": seed,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "learned_reward": learned["total_reward"],
        "learned_action_mean": learned["action_mean"],
        "learned_action_std": learned["action_std"],
        "learned_density_mean": learned["density_mean"],
        "learned_queue_final": learned["queue_final"],
        "u0_reward": u0["total_reward"],
        "u05_reward": u05["total_reward"],
        "u1_reward": u1["total_reward"],
        "beats_u0": int(learned["total_reward"] > u0["total_reward"]),
        "beats_u05": int(learned["total_reward"] > u05["total_reward"]),
        "beats_u1": int(learned["total_reward"] > u1["total_reward"]),
        "best_baseline_reward": best_baseline,
        "margin_over_best_baseline": (
            learned["total_reward"] - best_baseline
        )
        / abs(best_baseline),
    }


def _print_summary(rows: list[dict], betas: list[float]) -> None:
    print("\n========= sweep summary =========")
    print(
        f"{'beta':>6} | {'n':>3} | "
        f"{'learned_mean':>13} | {'learned_std':>11} | "
        f"{'wins_u0':>7} | {'wins_u05':>8} | {'wins_u1':>7} | "
        f"{'mean_margin':>12} | acceptance"
    )
    print("-" * 110)
    for beta in betas:
        beta_rows = [r for r in rows if abs(r["beta"] - beta) < 1e-9]
        if not beta_rows:
            continue
        n = len(beta_rows)
        learned_rewards = [r["learned_reward"] for r in beta_rows]
        mean_r = sum(learned_rewards) / n
        std_r = (
            sum((x - mean_r) ** 2 for x in learned_rewards) / n
        ) ** 0.5
        wins_u0 = sum(r["beats_u0"] for r in beta_rows)
        wins_u05 = sum(r["beats_u05"] for r in beta_rows)
        wins_u1 = sum(r["beats_u1"] for r in beta_rows)
        mean_margin = sum(r["margin_over_best_baseline"] for r in beta_rows) / n

        # Acceptance: 4/5 wins over u=1 AND mean margin >= 5%
        passes = wins_u1 >= max(1, int(0.8 * n)) and mean_margin >= 0.05
        verdict = "PASS" if passes else "fail"
        print(
            f"{beta:>6.2f} | {n:>3} | "
            f"{mean_r:>13.2f} | {std_r:>11.2f} | "
            f"{wins_u0:>5}/{n} | {wins_u05:>6}/{n} | {wins_u1:>5}/{n} | "
            f"{mean_margin:>+11.2%} | {verdict}"
        )
    print("=================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5b reward-weight sweep")
    parser.add_argument(
        "--betas",
        type=str,
        default="0.3,1.0,3.0",
        help="Comma-separated beta values",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override for smoke runs (default: use config's 100k)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl/ppo_surrogate.yaml",
        help="Base PPO config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dir (default: runs/ppo/m5b_sweep_<timestamp>)",
    )
    args = parser.parse_args()

    betas = _parse_csv_arg(args.betas, float)
    seeds = _parse_csv_arg(args.seeds, int)

    if args.output:
        sweep_dir = Path(args.output)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = PROJECT_ROOT / "runs" / "ppo" / f"m5b_sweep_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = sweep_dir / "results.csv"

    print(
        f"[sweep] betas={betas} seeds={seeds} "
        f"total_timesteps={args.total_timesteps or 'config-default'} "
        f"\n[sweep] sweep_dir={sweep_dir}"
    )

    rows: list[dict] = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for beta, seed in itertools.product(betas, seeds):
            run_dir = _train_one(beta, seed, args.config, args.total_timesteps, log_dir)
            row = _evaluate_run(run_dir, beta, seed)
            rows.append(row)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            print(
                f"[sweep] beta={beta} seed={seed} learned={row['learned_reward']:.1f} "
                f"u0={row['u0_reward']:.1f} u05={row['u05_reward']:.1f} "
                f"u1={row['u1_reward']:.1f} margin={row['margin_over_best_baseline']:+.2%}"
            )

    _print_summary(rows, betas)
    print(f"[sweep] results.csv → {csv_path}")


if __name__ == "__main__":
    main()
