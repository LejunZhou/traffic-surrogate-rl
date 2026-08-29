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

import gymnasium as gym
import numpy as np
import torch
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
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
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
    eval_env = None
    try:
        env = Monitor(_make_env(env_type, env_cfg), filename=str(run_dir / "monitor.csv"))
        callbacks = []
        total_timesteps = int(training_cfg["total_timesteps"])
        checkpoint_freq = int(training_cfg.get("checkpoint_freq", 0))
        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=f"ppo_{env_type}",
                )
            )
        eval_freq = int(training_cfg.get("eval_freq", 0))
        if eval_freq > 0:
            # Deterministic evaluation on a separate env instance; SB3 writes
            # best_model.zip to the run dir whenever eval/mean_reward improves.
            # The deployed policy is best_model.zip, not final_model.zip: on
            # SUMO the deterministic policy can drift while the stochastic
            # training return still improves (M7 runs 2-3).
            eval_base = _make_env(env_type, env_cfg)
            if bool(training_cfg.get("eval_cycle_cells", False)):
                eval_base = CycleDemandCells(
                    eval_base,
                    demands=[float(v) for v in env_cfg.get("demand_levels", [])] or [None],
                    ramps=[float(v) for v in env_cfg.get("ramp_demand_levels", [])] or [None],
                    base_seed=int(training_cfg.get("eval_seed", 10_000)),
                )
                print(f"[train_ppo] eval cycles through {len(eval_base.cells)} (mainline, ramp) cells with fixed seeds")
            eval_env = Monitor(eval_base)
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_dir),
                    log_path=str(run_dir / "eval"),
                    eval_freq=eval_freq,
                    n_eval_episodes=int(training_cfg.get("n_eval_episodes", 1)),
                    deterministic=True,
                    render=False,
                    verbose=1,
                )
            )
            print(f"[train_ppo] EvalCallback: deterministic eval every {eval_freq} steps -> best_model.zip")
        info_callback = _make_wandb_info_callback(
            wandb_run,
            log_freq=int(training_cfg.get("wandb_info_log_freq", 1)),
        )
        if info_callback is not None:
            callbacks.append(info_callback)
        entropy_callback = _make_entropy_coef_callback(
            ppo_cfg,
            total_timesteps=total_timesteps,
            wandb_run=wandb_run,
            log_freq=int(training_cfg.get("wandb_info_log_freq", 1)),
        )
        if entropy_callback is not None:
            callbacks.append(entropy_callback)

        ppo_kwargs = _ppo_kwargs(ppo_cfg)
        model = PPO(
            policy=str(ppo_cfg.get("policy", "MlpPolicy")),
            env=env,
            seed=seed,
            tensorboard_log=None,
            **ppo_kwargs,
        )
        init_u = training_cfg.get("action_init_u")
        if init_u is not None:
            _set_initial_action_mean(
                model, float(init_u), symmetric=bool(env_cfg.get("symmetric_action", False))
            )
        sb3_logger = configure(str(run_dir), ["stdout", "csv"])
        wandb_format = _make_wandb_output_format(wandb_run)
        if wandb_format is not None:
            sb3_logger.output_formats.append(wandb_format)
        model.set_logger(sb3_logger)

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
        if eval_env is not None:
            eval_env.close()
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


class CycleDemandCells(gym.Wrapper):
    """Deterministic evaluation schedule: successive resets walk through every
    (mainline demand, ramp demand) cell in order with a fixed SUMO seed per
    pass, so EvalCallback's mean over n_eval_episodes = n_cells compares the
    same scenarios every time instead of a random draw of cells."""

    def __init__(self, env, demands, ramps, base_seed: int):
        super().__init__(env)
        self.cells = [(d, r) for d in demands for r in ramps]
        self.base_seed = int(base_seed)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        d, r = self.cells[self._i % len(self.cells)]
        opts = dict(options or {})
        if d is not None:
            opts.setdefault("demand_vph", d)
        if r is not None:
            opts.setdefault("ramp_demand_vph", r)
        opts.setdefault("sumo_seed", self.base_seed + self._i // len(self.cells))
        self._i += 1
        return self.env.reset(seed=seed, options=opts)


def _set_initial_action_mean(model, u0: float, symmetric: bool) -> None:
    """Start the Gaussian policy at metering rate u0 instead of SB3's default.

    SB3 initialises the mean head (`policy.action_net`, a Linear layer) with
    gain 0.01 and zero bias, so the untrained policy outputs a ≈ 0 for every
    observation: u = 0.5 with env.symmetric_action, u = 0 (the clipped box
    corner) without. Neither is chosen for the traffic; this sets the bias so
    the initial deterministic action is u0 (read off the constant-u sweep),
    leaving the weights untouched — the policy still maps observations to
    actions from step one.
    """
    if not 0.0 <= u0 <= 1.0:
        raise ValueError(f"training.action_init_u must be in [0, 1], got {u0}")
    a0 = 2.0 * u0 - 1.0 if symmetric else u0
    action_net = getattr(model.policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "bias"):
        raise RuntimeError("action_init_u requires a Gaussian MlpPolicy with a Linear action_net")
    with torch.no_grad():
        action_net.bias.fill_(a0)
    print(f"[train_ppo] action_init_u={u0}: initial policy mean set to a={a0:+.3f} "
          f"({'symmetric' if symmetric else 'raw'} action space)")


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
    """Build the env and, if env.symmetric_action is true, expose PPO a
    symmetric action Box [-1, 1] that gymnasium rescales to the env's [0, 1].

    SB3 initialises the Gaussian policy mean at ~0 with std 1. On a [0, 1]
    Box that puts the initial *deterministic* action at u = 0 (ramp closed)
    and clips half of every sampled batch to 0, which is the u=0 collapse
    seen in M6 / M6b / M7 run 1. With the symmetric Box the initial mean maps
    to u = 0.5 and clipping is balanced. Evaluation scripts detect the
    trained model's action-space bounds and undo the mapping.
    """
    env = _make_base_env(env_type, env_cfg)
    if bool(env_cfg.get("symmetric_action", False)):
        import gymnasium as gym

        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        print("[train_ppo] symmetric_action=true: PPO acts in [-1, 1], rescaled to u in [0, 1]")
    return env


def _make_base_env(env_type: str, env_cfg: dict):
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


def _make_entropy_coef_callback(
    ppo_cfg: dict,
    *,
    total_timesteps: int,
    wandb_run,
    log_freq: int,
):
    schedule_cfg = ppo_cfg.get("ent_coef_schedule") or {}
    if not bool(schedule_cfg.get("enabled", False)):
        return None

    schedule_type = str(schedule_cfg.get("type", "exponential")).lower()
    supported_schedules = {"linear", "exponential", "power"}
    if schedule_type not in supported_schedules:
        valid = ", ".join(sorted(supported_schedules))
        raise ValueError(
            "ppo.ent_coef_schedule.type must be one of "
            f"{valid}; got {schedule_type!r}."
        )

    initial = float(schedule_cfg.get("initial", ppo_cfg.get("ent_coef", 0.0)))
    final = float(schedule_cfg.get("final", initial))
    if initial < 0.0 or final < 0.0:
        raise ValueError("Entropy coefficients must be non-negative.")
    if schedule_type == "exponential" and initial <= 0.0:
        raise ValueError("Exponential entropy schedule requires initial > 0.")
    if schedule_type == "exponential" and final <= 0.0:
        raise ValueError("Exponential entropy schedule requires final > 0.")
    power = float(schedule_cfg.get("power", 3.0))
    if power <= 0.0:
        raise ValueError("ppo.ent_coef_schedule.power must be positive.")
    total_timesteps = max(int(total_timesteps), 1)
    log_freq = max(int(log_freq), 1)

    from stable_baselines3.common.callbacks import BaseCallback

    class EntropyCoefScheduleCallback(BaseCallback):
        def _on_training_start(self) -> None:
            self._set_ent_coef(0)

        def _on_step(self) -> bool:
            self._set_ent_coef(self.num_timesteps)
            return True

        def _set_ent_coef(self, num_timesteps: int) -> None:
            progress = min(max(float(num_timesteps) / total_timesteps, 0.0), 1.0)
            ent_coef = _scheduled_entropy_coef(
                progress=progress,
                initial=initial,
                final=final,
                schedule_type=schedule_type,
                power=power,
            )
            self.model.ent_coef = ent_coef
            self.logger.record("train/ent_coef", ent_coef)
            if wandb_run is not None and num_timesteps % log_freq == 0:
                wandb_run.log({"train/ent_coef": ent_coef}, step=num_timesteps)

    return EntropyCoefScheduleCallback()


def _scheduled_entropy_coef(
    *,
    progress: float,
    initial: float,
    final: float,
    schedule_type: str,
    power: float,
) -> float:
    if schedule_type == "linear":
        fraction_remaining = 1.0 - progress
    elif schedule_type == "exponential":
        return float(initial * ((final / initial) ** progress))
    elif schedule_type == "power":
        fraction_remaining = (1.0 - progress) ** power
    else:
        raise ValueError(f"Unsupported entropy schedule type: {schedule_type!r}")
    return float(final + (initial - final) * fraction_remaining)


def _info_scalars(info: dict) -> dict[str, float]:
    keys = (
        "ramp_rate",
        "mean_density",
        "std_density",
        "outflow_vph",
        "lost_outflow_frac",
        "outflow_penalty",
        "queue_length",
        "queue_penalty",
        "std_penalty",
        "queue_norm",
        "queue_scale",
        "reward_delta",
        "reward_beta",
        "reward_gamma",
        "reward_q_ref",
        "reward_sigma_ref",
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
        "pending_mainline",
        "episode_pending_mainline_max",
        "discarded_mainline",
        "discarded_ramp",
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
        "--reward-delta",
        type=float,
        default=None,
        help="Override env.reward.delta (lost-outflow weight)",
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
        "--reward-q-ref",
        type=float,
        default=None,
        help="Override env.reward.q_ref (outflow reference, veh/h)",
    )
    parser.add_argument(
        "--reward-sigma-ref",
        type=float,
        default=None,
        help="Override env.reward.sigma_ref (density-std normaliser, veh/km)",
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
        "delta": args.reward_delta,
        "beta": args.reward_beta,
        "gamma": args.reward_gamma,
        "q_ref": args.reward_q_ref,
        "sigma_ref": args.reward_sigma_ref,
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
