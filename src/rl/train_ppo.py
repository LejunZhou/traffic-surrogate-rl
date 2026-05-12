"""
Train a PPO policy using Stable-Baselines3.

This entry point currently implements the direct SUMO+RL path only. It trains
PPO against rl.sumo_env_wrapper.SumoEnv and deliberately does not route through
the DeepONet surrogate environment.

Outputs (saved to a timestamped run directory):
- PPO model checkpoints
- Stable-Baselines3 monitor CSV
- Config snapshot and random seed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from utils.config import load_config
from utils.logging import make_run_dir


_PPO_ALLOWED_KEYS = {
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "use_sde",
    "sde_sample_freq",
    "target_kl",
    "device",
    "verbose",
}


def train(config: dict) -> None:
    """Train PPO on the configured environment.

    Args:
        config: Training config. Expected keys include:
            env.type ("surrogate" | "sumo"),
            env (environment-specific sub-config),
            ppo (SB3 PPO hyperparameters: n_steps, batch_size, n_epochs, lr, ...),
            training (total_timesteps, eval_freq, seed),
            output.run_dir
    """
    try:
        import torch
        from rl.sumo_env_wrapper import SumoEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "SUMO+RL training requires stable-baselines3, gymnasium, torch, and TraCI. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    env_cfg = dict(config["env"])
    env_type = env_cfg.get("type", "sumo")
    if env_type != "sumo":
        raise NotImplementedError(
            "Only direct SUMO+RL training is implemented. Set env.type: sumo."
        )

    training_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})
    output_cfg = config.get("output", {})
    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed, torch)

    project_root = Path(config.get("project_root", Path.cwd())).resolve()
    env_cfg.setdefault("project_root", str(project_root))

    base_dir = _resolve_path(output_cfg.get("base_dir", "runs/rl"), project_root)
    run_dir = make_run_dir(
        str(base_dir),
        output_cfg.get("run_name", "ppo_sumo"),
    )
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with (run_dir / "run_info.json").open("w") as f:
        json.dump({"seed": seed, "env_type": env_type}, f, indent=2)

    print(f"[train_ppo] Run dir: {run_dir}")
    print("[train_ppo] Backend: live SUMO via TraCI")

    env = Monitor(SumoEnv(env_cfg), filename=str(run_dir / "monitor.csv"))
    callbacks = []
    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 0))
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_sumo",
            )
        )

    ppo_kwargs = _ppo_kwargs(ppo_cfg)
    model = PPO(
        policy=str(ppo_cfg.get("policy", "MlpPolicy")),
        env=env,
        seed=seed,
        tensorboard_log=str(run_dir / "tb"),
        **ppo_kwargs,
    )

    total_timesteps = int(training_cfg["total_timesteps"])
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=bool(training_cfg.get("progress_bar", False)),
        )
        final_path = run_dir / "final_model"
        model.save(str(final_path))
        print(f"[train_ppo] Saved final policy to {final_path}.zip")
    finally:
        env.close()


def _ppo_kwargs(ppo_cfg: dict) -> dict:
    kwargs = {
        key: value
        for key, value in ppo_cfg.items()
        if key in _PPO_ALLOWED_KEYS
    }
    if "lr" in ppo_cfg and "learning_rate" not in kwargs:
        kwargs["learning_rate"] = ppo_cfg["lr"]
    kwargs.setdefault("verbose", 1)
    return kwargs


def _set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO directly in SUMO")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    cfg["project_root"] = str(project_root)
    train(cfg)


if __name__ == "__main__":
    main()
