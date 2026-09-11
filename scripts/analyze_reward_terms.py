"""
Compare per-step reward-term integrals across M5c (success) and M6 / M6b (fail).

Replays the saved PPO checkpoint for each milestone through its training env
(SurrogateEnv for M5c, SumoEnv for M6 / M6b), captures the per-step
`info["density_excess_penalty"]`, `info["queue_penalty"]`, `info["std_penalty"]`
already produced by the env, and writes:
- per-condition per_episode_terms.npz + summary.json
- a cross-condition stacked-bar plot of episode-mean integrals
- a 3-panel time-series plot (mean ± std across episodes)
- comparison_summary.json

The three runs all trained on identical reward weights
(alpha = beta = gamma = 1.0, rho_freeflow = 20.0, queue_norm = 100.0), so the
per-term magnitudes are directly comparable.

Usage:
    python scripts/analyze_reward_terms.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


CONDITIONS = [
    {
        "name": "m5c",
        "label": "M5c (surrogate, success)",
        "run_dir": "runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_20260512_021425",
        "checkpoint": "best_model.zip",
    },
    {
        "name": "m6",
        "label": "M6 (SUMO 20k, fail)",
        "run_dir": "runs/rl/ppo_sumo_constant_inflow_m6_seed0_20260512_023250",
        "checkpoint": "final_model.zip",
    },
    {
        "name": "m6b",
        "label": "M6b (SUMO 100k, fail)",
        "run_dir": "runs/rl/ppo_sumo_constant_inflow_m6b_seed0_100k_20260512_045505",
        "checkpoint": "final_model.zip",
    },
    {
        "name": "m6c",
        "label": "M6c (SUMO 20k, legacy, fail)",
        "run_dir": "runs/rl/ppo_sumo_constant_inflow_20260512_232240",
        "checkpoint": "final_model.zip",
    },
    {
        "name": "m6d",
        "label": "M6d (SUMO 20k, centered, fail)",
        "run_dir": "runs/rl/ppo_sumo_constant_inflow_20260513_004753",
        "checkpoint": "final_model.zip",
    },
]


def _find_latest_surrogate(surrogate_root: Path) -> Path | None:
    """Pick the most recent `deeponet_constant_inflow_*/best.pt`.

    Replacement for the missing rl.surrogate_env.find_latest_checkpoint
    helper that eval_constant_baselines.py expects. Sorts by the
    timestamp embedded in the directory name (format
    deeponet_constant_inflow_YYYYMMDD_HHMMSS), excluding *_smoke runs.
    """
    if not surrogate_root.exists():
        return None
    candidates = sorted(
        p for p in surrogate_root.glob("deeponet_constant_inflow_*/best.pt")
        if "smoke" not in p.parent.name
    )
    return candidates[-1] if candidates else None


def _load_run_env_config(run_dir: Path) -> dict:
    """Load env config from a run's saved config.yaml; resolve relative paths."""
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"])
    env_cfg["project_root"] = str(PROJECT_ROOT)

    sc = env_cfg.get("sumo_config")
    if sc and not Path(sc).is_absolute():
        env_cfg["sumo_config"] = str(PROJECT_ROOT / sc)

    if env_cfg.get("type") == "surrogate":
        ck = env_cfg.get("surrogate_checkpoint")
        if ck in (None, "auto", "latest"):
            latest = _find_latest_surrogate(PROJECT_ROOT / "runs" / "surrogate")
            if latest is None:
                raise FileNotFoundError(
                    "No surrogate checkpoint found under runs/surrogate/."
                )
            env_cfg["surrogate_checkpoint"] = str(latest)
        elif not Path(env_cfg["surrogate_checkpoint"]).is_absolute():
            env_cfg["surrogate_checkpoint"] = str(
                PROJECT_ROOT / env_cfg["surrogate_checkpoint"]
            )

    return env_cfg


def _build_env(env_cfg: dict):
    env_type = env_cfg.get("type", "sumo")
    if env_type == "surrogate":
        from rl.surrogate_env import SurrogateEnv

        return SurrogateEnv(
            surrogate_checkpoint=env_cfg["surrogate_checkpoint"],
            config=env_cfg,
        )
    if env_type == "sumo":
        from rl.sumo_env_wrapper import SumoEnv

        return SumoEnv(env_cfg)
    raise ValueError(f"Unknown env.type: {env_type!r}")


def _replay_policy(
    env_cfg: dict,
    model_path: Path,
    n_episodes: int,
    base_seed: int,
) -> dict:
    """Run n_episodes deterministic rollouts; collect per-step penalty terms."""
    from stable_baselines3 import PPO

    env = _build_env(env_cfg)
    model = PPO.load(str(model_path))
    t_ctrl = int(env.T_ctrl)

    shape = (n_episodes, t_ctrl)
    de_pens = np.zeros(shape, dtype=np.float32)
    q_pens = np.zeros(shape, dtype=np.float32)
    s_pens = np.zeros(shape, dtype=np.float32)
    actions = np.zeros(shape, dtype=np.float32)
    rewards = np.zeros(shape, dtype=np.float32)
    densities = np.zeros(shape, dtype=np.float32)
    queues = np.zeros(shape, dtype=np.float32)
    consistency_max_err = 0.0

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            for k in range(t_ctrl):
                action, _ = model.predict(obs, deterministic=True)
                action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                obs, reward, terminated, _, info = env.step(action_arr)
                actions[ep, k] = float(action_arr[0])
                rewards[ep, k] = float(reward)
                de_pens[ep, k] = float(info["density_excess_penalty"])
                q_pens[ep, k] = float(info["queue_penalty"])
                s_pens[ep, k] = float(info["std_penalty"])
                densities[ep, k] = float(info["mean_density"])
                queues[ep, k] = float(
                    info.get("queue_length", info.get("analytical_queue", 0.0))
                )

                pen_sum = de_pens[ep, k] + q_pens[ep, k] + s_pens[ep, k]
                err = abs(pen_sum + float(reward))
                if err > consistency_max_err:
                    consistency_max_err = err
                if terminated:
                    break
    finally:
        env.close()

    return {
        "density_excess_penalty": de_pens,
        "queue_penalty": q_pens,
        "std_penalty": s_pens,
        "action": actions,
        "reward": rewards,
        "mean_density": densities,
        "queue": queues,
        "consistency_max_err": float(consistency_max_err),
    }


def _summarize(traces: dict) -> dict:
    """Per-episode integrals + cross-episode mean/std."""
    summary = {
        "n_episodes": int(traces["action"].shape[0]),
        "n_steps_per_episode": int(traces["action"].shape[1]),
        "action_mean": float(traces["action"].mean()),
        "action_std": float(traces["action"].std()),
        "action_min": float(traces["action"].min()),
        "action_max": float(traces["action"].max()),
        "reward_total_mean": float(traces["reward"].sum(axis=1).mean()),
        "reward_total_std": float(traces["reward"].sum(axis=1).std()),
        "mean_density_avg": float(traces["mean_density"].mean()),
        "queue_mean": float(traces["queue"].mean()),
        "queue_max": float(traces["queue"].max()),
        "consistency_max_err": float(traces["consistency_max_err"]),
    }
    for term in ("density_excess_penalty", "queue_penalty", "std_penalty"):
        ep_integrals = traces[term].sum(axis=1)
        summary[f"{term}_integral_mean"] = float(ep_integrals.mean())
        summary[f"{term}_integral_std"] = float(ep_integrals.std())
    total = sum(
        summary[f"{t}_integral_mean"]
        for t in ("density_excess_penalty", "queue_penalty", "std_penalty")
    )
    summary["total_integral_mean"] = float(total)
    summary["queue_share"] = (
        summary["queue_penalty_integral_mean"] / total if total > 0 else 0.0
    )
    summary["density_excess_share"] = (
        summary["density_excess_penalty_integral_mean"] / total if total > 0 else 0.0
    )
    summary["std_share"] = (
        summary["std_penalty_integral_mean"] / total if total > 0 else 0.0
    )
    return summary


def _save_traces(traces: dict, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "per_episode_terms.npz",
        density_excess_penalty=traces["density_excess_penalty"],
        queue_penalty=traces["queue_penalty"],
        std_penalty=traces["std_penalty"],
        action=traces["action"],
        reward=traces["reward"],
        mean_density=traces["mean_density"],
        queue=traces["queue"],
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _plot_stacked_bar(results: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    x = np.arange(len(conds))
    width = 0.6

    terms = ["density_excess_penalty", "queue_penalty", "std_penalty"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    legend_labels = [
        "density excess (alpha * ReLU)",
        "queue (beta * quadratic)",
        "std(rho) (gamma * linear)",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottoms = np.zeros(len(conds))
    for term, c, lab in zip(terms, colors, legend_labels):
        vals = np.array(
            [results[cond]["summary"][f"{term}_integral_mean"] for cond in conds]
        )
        errs = np.array(
            [results[cond]["summary"][f"{term}_integral_std"] for cond in conds]
        )
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=width,
            color=c,
            label=lab,
            yerr=errs,
            ecolor="black",
            capsize=3,
        )
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels([results[c]["label"] for c in conds], rotation=15, ha="right")
    ax.set_ylabel("Episode-integrated penalty (sum over 120 steps, mean across episodes)")
    ax.set_title("Reward-term decomposition by training run")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_timeseries(results: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    terms = ["density_excess_penalty", "queue_penalty", "std_penalty"]
    titles = [
        "alpha * max(0, mean(rho) - 20)",
        "beta * (queue / 100)^2",
        "gamma * std(rho)",
    ]
    cond_colors = {
        "m5c": "#2ca02c",
        "m6": "#d62728",
        "m6b": "#9467bd",
        "m6c": "#8c564b",
        "m6d": "#e377c2",
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, term, title in zip(axes, terms, titles):
        for cond_name, cond_results in results.items():
            traces = cond_results["traces"][term]
            mean = traces.mean(axis=0)
            std = traces.std(axis=0)
            steps = np.arange(traces.shape[1])
            ax.plot(
                steps,
                mean,
                color=cond_colors.get(cond_name, "black"),
                label=cond_results["label"],
                linewidth=2,
            )
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=cond_colors.get(cond_name, "black"),
                alpha=0.15,
            )
        ax.set_title(title)
        ax.set_ylabel("Penalty per step")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Control step k (T_ctrl = 120)")
    fig.suptitle("Per-step reward-term traces by training run (mean +/- std)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reward-term integrals across M5c / M6 / M6b"
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write outputs. Defaults to runs/analysis/<ts>_reward_term_comparison.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "runs" / "analysis" / f"{timestamp}_reward_term_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze_reward_terms] Output dir: {output_dir}")

    results: dict = {}
    for cond in CONDITIONS:
        run_dir = PROJECT_ROOT / cond["run_dir"]
        model_path = run_dir / cond["checkpoint"]
        if not model_path.exists():
            print(f"[skip] Missing checkpoint: {model_path}")
            continue

        print(f"\n=== {cond['name']}: {cond['label']} ===")
        print(f"  run_dir   = {run_dir}")
        print(f"  checkpoint= {model_path}")
        env_cfg = _load_run_env_config(run_dir)
        if env_cfg.get("type") == "surrogate":
            print(f"  surrogate = {env_cfg['surrogate_checkpoint']}")

        t0 = time.perf_counter()
        traces = _replay_policy(env_cfg, model_path, args.n_episodes, args.base_seed)
        wall = time.perf_counter() - t0
        summary = _summarize(traces)
        summary["wall_clock_s"] = float(wall)
        summary["checkpoint"] = str(model_path)
        summary["env_type"] = env_cfg.get("type", "sumo")

        cond_out_dir = output_dir / cond["name"]
        _save_traces(traces, summary, cond_out_dir)

        print(
            f"  action      : mean={summary['action_mean']:.3f}  "
            f"std={summary['action_std']:.3f}  "
            f"[{summary['action_min']:.3f}, {summary['action_max']:.3f}]"
        )
        print(
            f"  reward total: mean={summary['reward_total_mean']:.2f}  "
            f"std={summary['reward_total_std']:.2f}"
        )
        print(
            f"  density-excess integral: {summary['density_excess_penalty_integral_mean']:.2f}"
            f" +/- {summary['density_excess_penalty_integral_std']:.2f}"
            f"  ({100 * summary['density_excess_share']:.1f}%)"
        )
        print(
            f"  queue integral         : {summary['queue_penalty_integral_mean']:.2f}"
            f" +/- {summary['queue_penalty_integral_std']:.2f}"
            f"  ({100 * summary['queue_share']:.1f}%)"
        )
        print(
            f"  std(rho) integral      : {summary['std_penalty_integral_mean']:.2f}"
            f" +/- {summary['std_penalty_integral_std']:.2f}"
            f"  ({100 * summary['std_share']:.1f}%)"
        )
        print(
            f"  consistency max err    : {summary['consistency_max_err']:.2e}"
            f"   (penalties_sum vs -reward)"
        )
        print(f"  wall_clock             : {wall:.1f}s")

        results[cond["name"]] = {
            "label": cond["label"],
            "summary": summary,
            "traces": traces,
        }

    if not results:
        print("[analyze_reward_terms] No conditions completed; exiting.")
        return

    _plot_stacked_bar(results, output_dir / "stacked_bar.png")
    _plot_timeseries(results, output_dir / "timeseries.png")
    print(
        f"\n[analyze_reward_terms] Plots: {output_dir / 'stacked_bar.png'}, "
        f"{output_dir / 'timeseries.png'}"
    )

    cross = {
        cond_name: {
            "label": r["label"],
            "action_mean": r["summary"]["action_mean"],
            "reward_total_mean": r["summary"]["reward_total_mean"],
            "density_excess_integral_mean": r["summary"][
                "density_excess_penalty_integral_mean"
            ],
            "queue_integral_mean": r["summary"]["queue_penalty_integral_mean"],
            "std_integral_mean": r["summary"]["std_penalty_integral_mean"],
            "total_integral_mean": r["summary"]["total_integral_mean"],
            "queue_share": r["summary"]["queue_share"],
            "density_excess_share": r["summary"]["density_excess_share"],
            "std_share": r["summary"]["std_share"],
        }
        for cond_name, r in results.items()
    }
    summary_path = output_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2)
    print(f"[analyze_reward_terms] Summary: {summary_path}")


if __name__ == "__main__":
    main()
