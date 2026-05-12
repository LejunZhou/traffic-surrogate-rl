"""
Train a PPO policy using Stable-Baselines3.

Dispatches on `env.type`:
- `env.type: "sumo"`  → direct SUMO+RL via rl.sumo_env_wrapper.SumoEnv
                       (the original direct path; wilson's setup).
- `env.type: "surrogate"` → surrogate-backed RL via rl.surrogate_env.SurrogateEnv
                          (Milestone 5; uses the DeepONet checkpoint
                          from Milestone 3).

Both backends expose identical observation, action, and reward contracts
so the SB3 wiring below is shared.

Outputs (saved to a timestamped run directory):
- PPO model checkpoints (best_model.zip + final_model.zip)
- Stable-Baselines3 monitor CSV (per-env, in monitor/)
- TensorBoard logs (tb/)
- Eval reward curve (evaluations.npz) when EvalCallback is used
- Config snapshot, random seed, wall-clock summary
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
            output.base_dir, output.run_name
    """
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "RL training requires stable-baselines3, gymnasium, torch, and (for "
            "env.type=sumo) TraCI. Install project dependencies with "
            "`pip install -e .`."
        ) from exc

    env_cfg = dict(config["env"])
    env_type = env_cfg.get("type", "sumo")
    if env_type not in ("sumo", "surrogate"):
        raise NotImplementedError(
            f"Unknown env.type={env_type!r}. Supported: 'sumo', 'surrogate'."
        )

    training_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})
    output_cfg = config.get("output", {})
    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed, torch)

    project_root = Path(config.get("project_root", Path.cwd())).resolve()
    env_cfg.setdefault("project_root", str(project_root))

    default_run_name = "ppo_sumo" if env_type == "sumo" else "ppo_surrogate"
    base_dir = _resolve_path(output_cfg.get("base_dir", "runs/rl"), project_root)
    run_dir = make_run_dir(
        str(base_dir),
        output_cfg.get("run_name", default_run_name),
    )
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "env_type": env_type}, f, indent=2)

    print(f"[train_ppo] Run dir: {run_dir}")
    print(
        f"[train_ppo] Backend: "
        f"{'live SUMO via TraCI' if env_type == 'sumo' else 'DeepONet surrogate'}"
    )

    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    train_env = _build_vec_env(env_type, env_cfg, monitor_dir, role="train", DummyVecEnv=DummyVecEnv, Monitor=Monitor)

    callbacks: list = []
    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 0))
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"ppo_{env_type}",
            )
        )

    # Eval callback: only enabled for the surrogate path by default.
    # The SUMO path is expensive (live simulation per eval) so wilson's
    # original setup omitted it; keep that default and only register when
    # explicitly requested via training.eval_freq > 0 (any env).
    eval_freq = int(training_cfg.get("eval_freq", 0))
    if eval_freq <= 0 and env_type == "surrogate":
        eval_freq = 4800  # default for surrogate: 10 rollouts
    eval_env = None
    if eval_freq > 0:
        eval_env = _build_vec_env(env_type, env_cfg, monitor_dir, role="eval", DummyVecEnv=DummyVecEnv, Monitor=Monitor)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=eval_freq,
                n_eval_episodes=int(training_cfg.get("n_eval_episodes", 5)),
                deterministic=True,
                render=False,
            )
        )

    ppo_kwargs = _ppo_kwargs(ppo_cfg)
    model = PPO(
        policy=str(ppo_cfg.get("policy", "MlpPolicy")),
        env=train_env,
        seed=seed,
        tensorboard_log=str(run_dir / "tb"),
        **ppo_kwargs,
    )

    total_timesteps = int(training_cfg["total_timesteps"])
    t0 = time.perf_counter()
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
        wall_clock_s = time.perf_counter() - t0
        with (run_dir / "wall_clock.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "wall_clock_s": wall_clock_s,
                    "total_timesteps": total_timesteps,
                    "timesteps_per_sec": (
                        float(total_timesteps) / wall_clock_s
                        if wall_clock_s > 0
                        else None
                    ),
                },
                f,
                indent=2,
            )
        train_env.close()
        if eval_env is not None:
            eval_env.close()


def _build_env(env_type: str, env_cfg: dict):
    """Construct a single (un-wrapped) environment instance from config."""
    if env_type == "sumo":
        from rl.sumo_env_wrapper import SumoEnv

        return SumoEnv(env_cfg)
    if env_type == "surrogate":
        from rl.surrogate_env import SurrogateEnv, find_latest_checkpoint

        ckpt_path = env_cfg["surrogate_checkpoint"]
        if ckpt_path in ("auto", "latest"):
            project_root = Path(env_cfg.get("project_root", Path.cwd())).resolve()
            latest = find_latest_checkpoint(project_root / "runs" / "surrogate")
            if latest is None:
                raise FileNotFoundError(
                    "env.surrogate_checkpoint='auto' but no checkpoint found "
                    "under runs/surrogate/. Train one with `python -m "
                    "surrogate.train --config configs/surrogate/baseline.yaml`."
                )
            ckpt_path = str(latest)
        return SurrogateEnv(
            surrogate_checkpoint=ckpt_path,
            config=env_cfg,
        )
    raise ValueError(f"Unknown env.type: {env_type!r}")


def _build_vec_env(
    env_type: str,
    env_cfg: dict,
    monitor_dir: Path,
    role: str,
    *,
    DummyVecEnv,
    Monitor,
):
    """Wrap a single env with Monitor + DummyVecEnv for SB3 consumption."""

    def _thunk():
        env = _build_env(env_type, env_cfg)
        return Monitor(env, filename=str(monitor_dir / f"{role}.csv"))

    return DummyVecEnv([_thunk])


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
    parser = argparse.ArgumentParser(description="Train PPO (SUMO or surrogate env)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps for smoke runs",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    cfg["project_root"] = str(project_root)
    if args.total_timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(args.total_timesteps)
    train(cfg)


if __name__ == "__main__":
    main()
