"""Train PPO in live SUMO or the batched M14 GRU DeepONet ensemble.

Use configs/ppo.yaml with configs/env_sumo.yaml or configs/env_surrogate.yaml.
Evaluation cycles over frozen validation profiles; aggregation can warm-start
from an existing PPO checkpoint. Outputs include policies, CSV logs, a config
snapshot, and optional Weights & Biases metrics.
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

from utils.config import load_config, merge_configs
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

    project_root = Path(config.get("project_root", Path(__file__).resolve().parents[2])).resolve()
    env_cfg.setdefault("project_root", str(project_root))

    base_dir = _resolve_path(output_cfg.get("base_dir", "runs/rl"), project_root)
    if output_cfg.get("run_dir"):
        run_dir = _resolve_path(output_cfg["run_dir"], project_root)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
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
        env_cfg = _resolve_auto_normalisation(env_cfg, project_root)
        vec_mode = _is_vec_surrogate(env_type, env_cfg)
        ledger_study = training_cfg.get("ledger_study")
        ledger_purpose = str(training_cfg.get("ledger_purpose", "direct_ppo"))
        if vec_mode:
            from stable_baselines3.common.vec_env import VecMonitor

            env = VecMonitor(_make_env(env_type, env_cfg), filename=str(run_dir / "monitor.csv"))
            n_envs = int(env.num_envs)
        else:
            base = _make_env(env_type, env_cfg)
            if env_type == "sumo" and ledger_study:
                base = LedgerEpisodes(base, ledger_study, ledger_purpose, project_root, int(training_cfg.get("ledger_round", 0)))
            env = Monitor(base, filename=str(run_dir / "monitor.csv"))
            n_envs = 1
        callbacks = []
        total_timesteps = int(training_cfg["total_timesteps"])
        eval_freq = int(training_cfg.get("eval_freq", 0))
        checkpoint_freq = int(training_cfg.get("checkpoint_freq", 0))
        if bool(training_cfg.get("checkpoint_every_eval", False)) and eval_freq > 0:
            checkpoint_freq = eval_freq
        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(checkpoint_freq // n_envs, 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=f"ppo_{env_type}",
                )
            )
        if eval_freq > 0:
            # Deterministic evaluation on a separate env instance; SB3 writes
            # best_model.zip to the run dir whenever eval/mean_reward improves.
            # The deployed policy is best_model.zip, not final_model.zip: on
            # SUMO the deterministic policy can drift while the stochastic
            # training return still improves (M7 runs 2-3).
            n_eval_episodes = int(training_cfg.get("n_eval_episodes", 1))
            eval_profiles = training_cfg.get("eval_profiles")
            if vec_mode:
                eval_cfg = dict(env_cfg)
                eval_cfg["ensemble_mode"] = str(training_cfg.get("eval_ensemble_mode", "mean"))
                if eval_profiles:
                    eval_cfg["profiles"] = {"set": eval_profiles}
                    from sumo_env.demand_profiles import load_profile_set

                    n_eval_episodes = len(load_profile_set(_resolve_path(eval_profiles, project_root)))
                eval_cfg["n_envs"] = n_eval_episodes
                eval_cfg["seed"] = int(training_cfg.get("eval_seed", 10_000))
                eval_env = _make_env(env_type, eval_cfg)
                print(f"[train_ppo] eval: SurrogateVecEnv ({eval_cfg['ensemble_mode']}) over {n_eval_episodes} fixed profiles")
            else:
                eval_base = _make_env(env_type, env_cfg)
                if eval_profiles:
                    from sumo_env.demand_profiles import load_profile_set

                    profiles = load_profile_set(_resolve_path(eval_profiles, project_root))
                    eval_base = CycleProfiles(eval_base, profiles, base_seed=int(training_cfg.get("eval_seed", 10_000)))
                    n_eval_episodes = len(profiles)
                    print(f"[train_ppo] eval cycles through {len(profiles)} fixed profiles ({eval_profiles})")
                if env_type == "sumo" and ledger_study:
                    eval_base = LedgerEpisodes(eval_base, ledger_study, "eval_val", project_root, int(training_cfg.get("ledger_round", 0)))
                elif bool(training_cfg.get("eval_cycle_cells", False)):
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
                    eval_freq=max(eval_freq // n_envs, 1),
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=1,
                )
            )
            print(f"[train_ppo] EvalCallback: deterministic eval every {eval_freq} steps ({n_eval_episodes} episodes) -> best_model.zip")
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
        init_policy = training_cfg.get("init_policy")
        if init_policy:
            init_path = _resolve_path(init_policy, project_root)
            load_kwargs = {k: v for k, v in ppo_kwargs.items() if k != "policy_kwargs"}
            model = PPO.load(str(init_path), env=env, seed=seed, tensorboard_log=None, **load_kwargs)
            log_std_reset = training_cfg.get("log_std_reset")
            if log_std_reset is not None and hasattr(model.policy, "log_std"):
                with torch.no_grad():
                    model.policy.log_std.fill_(float(log_std_reset))
            print(f"[train_ppo] warm start from {init_path} (value net kept, lr {model.learning_rate}, "
                  f"target_kl {model.target_kl}, log_std reset {log_std_reset})")
        else:
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


class LedgerEpisodes(gym.Wrapper):
    """Write one ledger line per finished SUMO episode (purpose direct_ppo /
    finetune / eval_val), with the episode return and the breakdown flag
    (max detector density > 60 veh/km for >= 5 consecutive minutes)."""

    def __init__(self, env, study: str, purpose: str, project_root: Path, round_index: int = 0):
        super().__init__(env)
        from utils.ledger import Ledger

        self.ledger = Ledger(study, project_root)
        self.purpose = purpose
        self.round_index = int(round_index)
        self._ret = 0.0; self._jam_run = 0; self._breakdown = False; self._t0 = 0.0; self._policy = "training"

    def reset(self, *, seed=None, options=None):
        import time as _time

        self._ret, self._jam_run, self._breakdown, self._t0 = 0.0, 0, False, _time.time()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        import time as _time

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        dens = info.get("density")
        if dens is not None:
            self._jam_run = self._jam_run + 1 if float(np.max(dens)) > 60.0 else 0
            if self._jam_run >= 10:
                self._breakdown = True
        if terminated or truncated:
            self.ledger.log(self.round_index, self.purpose, str(info.get("profile_set", "grid")),
                            int(info.get("profile_index", -1)), int(getattr(self.env.unwrapped, "current_sumo_seed", -1)),
                            self._policy, self._ret, self._breakdown, _time.time() - self._t0)
        return obs, reward, terminated, truncated, info


class CycleProfiles(gym.Wrapper):
    """Deterministic evaluation on a frozen profile set: successive resets walk
    through the profiles in order with each profile's first SUMO seed (or
    base_seed + index), so EvalCallback's mean over n_eval_episodes =
    len(profiles) compares identical episodes every pass."""

    def __init__(self, env, profiles, base_seed: int = 10_000):
        super().__init__(env)
        self.profiles = list(profiles)
        self.base_seed = int(base_seed)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        p = self.profiles[self._i % len(self.profiles)]
        opts = dict(options or {})
        opts.setdefault("profile", p)
        opts.setdefault("sumo_seed", p.sumo_seeds[0] if p.sumo_seeds else self.base_seed + self._i % len(self.profiles))
        self._i += 1
        return self.env.reset(seed=seed, options=opts)


def _is_vec_surrogate(env_type: str, env_cfg: dict) -> bool:
    return env_type == "surrogate" and bool(env_cfg.get("ensemble_dir"))


def _resolve_auto_normalisation(env_cfg: dict, project_root: Path) -> dict:
    """density_mean/std: 'auto' -> the ensemble manifest (surrogate) or the
    rollout-store metadata (env.density_stats_from), so both envs z-score
    with the same numbers."""
    cfg = dict(env_cfg)
    if cfg.get("density_mean") not in ("auto", None) and cfg.get("density_std") not in ("auto", None):
        return cfg
    src = None
    if cfg.get("ensemble_dir"):
        src = _resolve_path(cfg["ensemble_dir"], project_root) / "manifest.json"
        key = "normalization"
    elif cfg.get("density_stats_from"):
        src = _resolve_path(cfg["density_stats_from"], project_root)
        key = None
    if src is None or not Path(src).exists():
        if cfg.get("density_mean") in ("auto", None):
            raise ValueError("env.density_mean is 'auto' but neither env.ensemble_dir/manifest.json nor env.density_stats_from exists")
        return cfg
    data = json.loads(Path(src).read_text())
    stats = data[key] if key else data.get("metadata", data)
    cfg["density_mean"] = float(stats["mean_density"])
    cfg["density_std"] = float(stats["std_density"])
    print(f"[train_ppo] density normalisation from {src}: mean {cfg['density_mean']:.3f} std {cfg['density_std']:.3f}")
    return cfg


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
    if _is_vec_surrogate(env_type, env_cfg):
        from rl.surrogate_vec_env import SurrogateVecEnv

        return SurrogateVecEnv(env_cfg)   # handles symmetric_action itself
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
        raise ValueError("M14 surrogate PPO requires env.ensemble_dir and the batched SurrogateVecEnv")

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


def _nested(keys: list[str], value):
    out = value
    for k in reversed(keys):
        out = {k: out}
    return out


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
    parser.add_argument("--config", default="configs/ppo.yaml", help="Path to YAML config")
    parser.add_argument("--overlay", action="append", default=[],
                        help="YAML overlay(s) deep-merged onto --config (e.g. configs/env_surrogate.yaml)")
    parser.add_argument("--init-policy", default=None, help="SB3 .zip to warm-start from (training.init_policy)")
    parser.add_argument("--run-name", default=None, help="override output.run_name")
    parser.add_argument("--ensemble-dir", default=None, help="override env.ensemble_dir")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted override, e.g. --set training.total_timesteps=1000000 --set env.n_envs=16")
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
    for ov in args.overlay:
        cfg = merge_configs(cfg, load_config(str(project_root / ov)))
    for spec in args.set:
        key, _, raw = spec.partition("=")
        cfg = merge_configs(cfg, _nested(key.split("."), yaml.safe_load(raw)))
    cfg["project_root"] = str(project_root)
    if args.init_policy:
        cfg.setdefault("training", {})["init_policy"] = args.init_policy
    if args.ensemble_dir:
        cfg.setdefault("env", {})["ensemble_dir"] = args.ensemble_dir
    if args.run_name:
        cfg.setdefault("output", {})["run_name"] = args.run_name
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
