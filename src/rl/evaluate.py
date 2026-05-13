"""
Evaluate a trained PPO policy in SUMO.

Always evaluates in SUMO (the ground-truth simulator).

Outputs:
- Per-episode metrics: total reward, mean density, throughput, queue length
- Trajectory plots for qualitative analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from utils.config import load_config


def evaluate_in_sumo(policy_path: str, config: dict) -> dict:
    """Roll out a trained PPO policy in SUMO for n_episodes and collect metrics.

    Args:
        policy_path: Path to a saved SB3 model (.zip).
        config: Evaluation config (sumo env config, n_episodes, output_dir, etc.).

    Returns:
        Dict of metrics:
            "mean_total_reward": float
            "mean_density":      float   (lower = less congestion)
            "throughput":        float   (veh/hr exiting the segment)
            "mean_queue_length": float   (on-ramp queue)
            "episodes":          list[dict]  (per-episode breakdown)
    """
    try:
        from rl.sumo_env_wrapper import SumoEnv
        from stable_baselines3 import PPO
        from utils.plotting import plot_control_sequence, plot_trajectory
    except ImportError as exc:
        raise ImportError(
            "SUMO policy evaluation requires stable-baselines3, gymnasium, torch, and TraCI. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    project_root = Path(config.get("project_root", Path.cwd())).resolve()
    env_cfg = _sumo_eval_env_config(config, project_root)
    env_cfg.setdefault("project_root", str(project_root))
    eval_cfg = config.get("evaluation", {})
    output_dir = _resolve_path(
        eval_cfg.get("output_dir", "runs/rl_eval"),
        project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    n_episodes = int(eval_cfg.get("n_episodes", 3))
    deterministic = bool(eval_cfg.get("deterministic", True))
    base_seed = int(eval_cfg.get("seed", config.get("training", {}).get("seed", 42)))
    save_plots = bool(eval_cfg.get("save_plots", True))
    max_plot_episodes = int(eval_cfg.get("max_plot_episodes", 3))

    env = SumoEnv(env_cfg)
    model = PPO.load(str(_resolve_path(policy_path, project_root)), env=None)
    episodes: list[dict] = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=base_seed + ep)
            done = False
            rewards: list[float] = []
            density_rows: list[np.ndarray] = []
            speed_rows: list[np.ndarray] = []
            flow_rows: list[np.ndarray] = []
            actions: list[float] = []
            queue_rows: list[float] = []
            physical_ramp_rows: list[float] = []
            last_info = info

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, last_info = env.step(action)
                done = bool(terminated or truncated)

                rewards.append(float(reward))
                actions.append(float(np.asarray(action).reshape(-1)[0]))
                density_rows.append(last_info["density"])
                speed_rows.append(last_info["speed"])
                flow_rows.append(last_info["flow"])
                queue_rows.append(float(last_info.get("queue_length", 0.0)))
                physical_ramp_rows.append(
                    float(last_info.get("interval_physical_ramp_mean", 0.0))
                )

            density = np.stack(density_rows, axis=1)
            speed = np.stack(speed_rows, axis=1)
            flow = np.stack(flow_rows, axis=1)
            episode = {
                "episode": ep,
                "total_reward": float(np.sum(rewards)),
                "mean_density": float(np.mean(density)),
                "max_density": float(np.max(density)),
                "mean_speed": float(np.mean(speed)),
                "mean_flow": float(np.mean(flow)),
                "throughput_vph": float(last_info.get("throughput_vph", 0.0)),
                "mean_queue_length": float(np.mean(queue_rows)) if queue_rows else 0.0,
                "max_queue_length": float(max(queue_rows)) if queue_rows else 0.0,
                "mean_physical_ramp_occupancy": float(np.mean(physical_ramp_rows))
                if physical_ramp_rows
                else 0.0,
                "max_physical_ramp_occupancy": float(max(physical_ramp_rows))
                if physical_ramp_rows
                else 0.0,
                "teleports": int(last_info.get("teleports", 0)),
                "insert_success": int(last_info.get("insert_success", 0)),
                "insert_attempts": int(last_info.get("insert_attempts", 0)),
                "demand_vph": float(last_info.get("demand_vph", 0.0)),
                "mean_action": float(np.mean(actions)),
            }
            episodes.append(episode)

            if save_plots and ep < max_plot_episodes:
                t_grid = env.t_grid[: density.shape[1]]
                plot_trajectory(
                    density=density,
                    x_grid=env.x_grid,
                    t_grid=t_grid,
                    output_path=output_dir / f"episode_{ep:03d}_density.png",
                    title=(
                        f"SUMO PPO rollout - episode {ep}, "
                        f"demand={episode['demand_vph']:.0f} vph"
                    ),
                )
                plot_control_sequence(
                    actions=np.asarray(actions, dtype=np.float32),
                    t_grid=t_grid,
                    output_path=output_dir / f"episode_{ep:03d}_control.png",
                    title=(
                        f"PPO ramp control - episode {ep}, "
                        f"demand={episode['demand_vph']:.0f} vph"
                    ),
                )

            np.savez(
                output_dir / f"episode_{ep:03d}_rollout.npz",
                density=density.astype(np.float32),
                speed=speed.astype(np.float32),
                flow=flow.astype(np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                queue_length=np.asarray(queue_rows, dtype=np.float32),
                physical_ramp_occupancy=np.asarray(
                    physical_ramp_rows,
                    dtype=np.float32,
                ),
                x_grid=env.x_grid.astype(np.float32),
                t_grid=env.t_grid[: density.shape[1]].astype(np.float32),
                demand_vph=np.array(episode["demand_vph"], dtype=np.float32),
            )

            print(
                "[eval_sumo] "
                f"ep={ep} reward={episode['total_reward']:.2f} "
                f"mean_density={episode['mean_density']:.2f} "
                f"throughput={episode['throughput_vph']:.0f} "
                f"queue={episode['mean_queue_length']:.1f} "
                f"physical_ramp={episode['mean_physical_ramp_occupancy']:.1f}"
            )
    finally:
        env.close()

    metrics = {
        "mean_total_reward": float(np.mean([e["total_reward"] for e in episodes])),
        "mean_density": float(np.mean([e["mean_density"] for e in episodes])),
        "throughput": float(np.mean([e["throughput_vph"] for e in episodes])),
        "mean_queue_length": float(np.mean([e["mean_queue_length"] for e in episodes])),
        "mean_physical_ramp_occupancy": float(
            np.mean([e["mean_physical_ramp_occupancy"] for e in episodes])
        ),
        "mean_teleports": float(np.mean([e["teleports"] for e in episodes])),
        "n_episodes": n_episodes,
        "episodes": episodes,
    }
    with (output_dir / "eval_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_sumo] Saved metrics to {output_dir / 'eval_metrics.json'}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes"}, indent=2))
    return metrics


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def _sumo_eval_env_config(config: dict, project_root: Path) -> dict:
    env_cfg = dict(config["env"])
    env_type = env_cfg.get("type", "sumo")
    if env_type not in {"sumo", "surrogate"}:
        raise ValueError("Evaluation config env.type must be 'sumo' or 'surrogate'.")

    if env_type == "surrogate":
        env_cfg["type"] = "sumo"
        if "density_mean" not in env_cfg or "density_std" not in env_cfg:
            normalization = _load_surrogate_normalization(env_cfg, project_root)
            env_cfg.setdefault("density_mean", normalization["mean_density"])
            env_cfg.setdefault("density_std", normalization["std_density"])

    return env_cfg


def _load_surrogate_normalization(env_cfg: dict, project_root: Path) -> dict:
    checkpoint_path = env_cfg.get("surrogate_checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Evaluating a surrogate-trained policy in SUMO requires either "
            "env.density_mean/env.density_std or env.surrogate_checkpoint so "
            "the SUMO observations use the same normalization as training."
        )

    import torch

    checkpoint = torch.load(
        str(_resolve_path(checkpoint_path, project_root)),
        map_location="cpu",
    )
    normalization = checkpoint.get("normalization")
    if not normalization:
        raise KeyError(
            f"Surrogate checkpoint {checkpoint_path!r} does not contain normalization."
        )
    return normalization


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy in SUMO")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--policy",
        default=None,
        help="Path to PPO .zip model. Overrides evaluation.policy_path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    cfg["project_root"] = str(project_root)
    eval_cfg = cfg.get("evaluation", {})
    policy_path = args.policy or eval_cfg.get("policy_path")
    if policy_path is None:
        raise ValueError(
            "Provide --policy or set evaluation.policy_path in the config."
        )
    evaluate_in_sumo(policy_path, cfg)


if __name__ == "__main__":
    main()
