"""
Train a PPO policy using Stable-Baselines3.

This entry point supports both PPO backends used in the Phase 1 pipeline:
- env.type=sumo: live SUMO through TraCI
- env.type=surrogate: trained DeepONet surrogate environment

Outputs (saved to a timestamped run directory):
- PPO model checkpoints
- Stable-Baselines3 monitor CSV
- Stable-Baselines3 progress CSV
- Weights & Biases metrics, when output.wandb.enabled=true
- Config snapshot and random seed
"""

from __future__ import annotations

import argparse
import copy
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
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.logger import configure
    except ImportError as exc:
        raise ImportError(
            "PPO training requires stable-baselines3, gymnasium, and torch. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    env_cfg = dict(config["env"])
    env_type = env_cfg.get("type", "sumo")
    if env_type not in {"sumo", "surrogate"}:
        raise ValueError("env.type must be either 'sumo' or 'surrogate'.")

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
        output_cfg.get("run_name", _default_run_name(env_type)),
    )
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with (run_dir / "run_info.json").open("w") as f:
        json.dump({"seed": seed, "env_type": env_type}, f, indent=2)

    wandb_run = _init_wandb(output_cfg, run_dir, config)

    print(f"[train_ppo] Run dir: {run_dir}")
    print(f"[train_ppo] Backend: {_backend_label(env_type)}")

    env = None
    try:
        env = Monitor(_make_env(env_type, env_cfg), filename=str(run_dir / "monitor.csv"))
        callbacks = []
        checkpoint_freq = int(training_cfg.get("checkpoint_freq", 0))
        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=f"ppo_{env_type}",
                )
            )
        info_callback = _make_wandb_info_callback(
            wandb_run,
            log_freq=int(training_cfg.get("wandb_info_log_freq", 1)),
        )
        if info_callback is not None:
            callbacks.append(info_callback)

        ppo_kwargs = _ppo_kwargs(ppo_cfg)
        model = PPO(
            policy=str(ppo_cfg.get("policy", "MlpPolicy")),
            env=env,
            seed=seed,
            tensorboard_log=None,
            **ppo_kwargs,
        )
        sb3_logger = configure(str(run_dir), ["stdout", "csv"])
        wandb_format = _make_wandb_output_format(wandb_run)
        if wandb_format is not None:
            sb3_logger.output_formats.append(wandb_format)
        model.set_logger(sb3_logger)

        total_timesteps = int(training_cfg["total_timesteps"])
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=bool(training_cfg.get("progress_bar", False)),
        )
        final_path = run_dir / "final_model"
        model.save(str(final_path))
        print(f"[train_ppo] Saved final policy to {final_path}.zip")
    finally:
        if env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()


def _ppo_kwargs(ppo_cfg: dict) -> dict:
    kwargs = {
        key: value
        for key, value in ppo_cfg.items()
        if key in _PPO_ALLOWED_KEYS
    }
    if "lr" in ppo_cfg and "learning_rate" not in kwargs:
        kwargs["learning_rate"] = ppo_cfg["lr"]
    if "policy_kwargs" in ppo_cfg:
        kwargs["policy_kwargs"] = _policy_kwargs(ppo_cfg["policy_kwargs"])
    kwargs.setdefault("verbose", 1)
    return kwargs


def _policy_kwargs(policy_kwargs_cfg: dict) -> dict:
    if not isinstance(policy_kwargs_cfg, dict):
        raise ValueError("ppo.policy_kwargs must be a mapping.")

    policy_kwargs = copy.deepcopy(policy_kwargs_cfg)
    activation_fn = policy_kwargs.get("activation_fn")
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = _activation_fn(activation_fn)
    return policy_kwargs


def _activation_fn(name_or_cls):
    if not isinstance(name_or_cls, str):
        return name_or_cls

    import torch.nn as nn

    key = name_or_cls.removeprefix("nn.").lower()
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if key not in activations:
        valid = ", ".join(sorted(activations))
        raise ValueError(
            f"Unsupported ppo.policy_kwargs.activation_fn={name_or_cls!r}. "
            f"Valid options: {valid}."
        )
    return activations[key]


def _make_env(env_type: str, env_cfg: dict):
    if env_type == "sumo":
        try:
            from rl.sumo_env_wrapper import SumoEnv
        except ImportError as exc:
            raise ImportError(
                "env.type=sumo requires TraCI/SUMO Python bindings. "
                "Set SUMO_HOME/PYTHONPATH or install SUMO before training "
                "against live SUMO."
            ) from exc
        return SumoEnv(env_cfg)

    if env_type == "surrogate":
        from rl.surrogate_env import SurrogateEnv

        return SurrogateEnv(config=env_cfg)

    raise ValueError(f"Unsupported env.type: {env_type!r}")


def _init_wandb(output_cfg: dict, run_dir: Path, config: dict):
    wandb_cfg = output_cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "PPO W&B logging is enabled, but the 'wandb' package is not installed. "
            "Install it with `pip install wandb` or set output.wandb.enabled=false."
        ) from exc

    return wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name") or run_dir.name,
        config=config,
        dir=str(run_dir),
        tags=wandb_cfg.get("tags"),
        mode=wandb_cfg.get("mode"),
    )


def _make_wandb_output_format(wandb_run):
    if wandb_run is None:
        return None

    from stable_baselines3.common.logger import KVWriter

    class WandbOutputFormat(KVWriter):
        def write(self, key_values, key_excluded, step: int = 0) -> None:
            metrics = {}
            for key, value in key_values.items():
                excluded = key_excluded.get(key, ())
                if "wandb" in excluded:
                    continue
                scalar = _as_scalar(value)
                if scalar is not None:
                    metrics[key] = scalar
            if metrics:
                wandb_run.log(metrics, step=step)

        def close(self) -> None:
            pass

    return WandbOutputFormat()


def _make_wandb_info_callback(wandb_run, log_freq: int):
    if wandb_run is None:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class WandbInfoCallback(BaseCallback):
        def __init__(self, log_freq_steps: int) -> None:
            super().__init__()
            self.log_freq_steps = max(int(log_freq_steps), 1)

        def _on_step(self) -> bool:
            if self.num_timesteps % self.log_freq_steps != 0:
                return True
            infos = self.locals.get("infos", [])
            metrics = {}
            for info in infos:
                metrics.update(_info_scalars(info))
            if metrics:
                wandb_run.log(metrics, step=self.num_timesteps)
            return True

    return WandbInfoCallback(log_freq)


def _info_scalars(info: dict) -> dict[str, float]:
    keys = (
        "ramp_rate",
        "mean_density",
        "std_density",
        "density_excess",
        "density_excess_penalty",
        "queue_length",
        "queue_penalty",
        "std_penalty",
        "queue_norm",
        "queue_scale",
        "reward_alpha",
        "reward_beta",
        "reward_gamma",
        "reward_rho_freeflow",
        "reward_queue_norm",
        "raw_reward",
        "reward_warmup_active",
        "reward_warmup_s",
        "episode_queue_mean",
        "episode_queue_max",
        "ramp_arrivals",
        "ramp_released",
        "ramp_release_capacity",
        "interval_physical_ramp_mean",
        "interval_physical_ramp_max",
        "episode_physical_ramp_mean",
        "episode_physical_ramp_max",
        "throughput_vph",
        "mean_speed",
        "mean_flow",
        "teleports",
        "insert_success",
        "insert_attempts",
        "insert_rejected",
    )
    metrics = {}
    for key in keys:
        if key not in info:
            continue
        scalar = _as_scalar(info[key])
        if scalar is not None:
            metrics[f"env/{key}"] = scalar
    return metrics


def _as_scalar(value) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return None


def _default_run_name(env_type: str) -> str:
    return "ppo_surrogate" if env_type == "surrogate" else "ppo_sumo"


def _backend_label(env_type: str) -> str:
    if env_type == "surrogate":
        return "DeepONet surrogate"
    return "live SUMO via TraCI"


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
    parser = argparse.ArgumentParser(description="Train a PPO policy")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps for smoke runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed",
    )
    parser.add_argument(
        "--reward-alpha",
        type=float,
        default=None,
        help="Override env.reward.alpha",
    )
    parser.add_argument(
        "--reward-beta",
        type=float,
        default=None,
        help="Override env.reward.beta",
    )
    parser.add_argument(
        "--reward-gamma",
        type=float,
        default=None,
        help="Override env.reward.gamma",
    )
    parser.add_argument(
        "--reward-rho-freeflow",
        type=float,
        default=None,
        help="Override env.reward.rho_freeflow",
    )
    parser.add_argument(
        "--reward-queue-norm",
        type=float,
        default=None,
        help="Override env.reward.queue_norm",
    )
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default=None,
        help="Append a suffix to output.run_name",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    cfg["project_root"] = str(project_root)
    if args.total_timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(args.total_timesteps)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = int(args.seed)

    reward_overrides = {
        "alpha": args.reward_alpha,
        "beta": args.reward_beta,
        "gamma": args.reward_gamma,
        "rho_freeflow": args.reward_rho_freeflow,
        "queue_norm": args.reward_queue_norm,
    }
    if any(value is not None for value in reward_overrides.values()):
        env_cfg = cfg.setdefault("env", {})
        reward_cfg = dict(env_cfg.get("reward") or {})
        for key, value in reward_overrides.items():
            if value is not None:
                reward_cfg[key] = float(value)
        env_cfg["reward"] = reward_cfg

    if args.run_name_suffix:
        out_cfg = cfg.setdefault("output", {})
        base_name = out_cfg.get("run_name", "ppo")
        out_cfg["run_name"] = f"{base_name}_{args.run_name_suffix}"
    train(cfg)


if __name__ == "__main__":
    main()
