"""
Evaluate a learned PPO policy or a constant ramp-metering policy on
SurrogateEnv, with configurable reward weights.

Used by:
- One-off comparisons of learned vs constant policies.
- The M5b sweep driver to score each run against u={0.0, 0.5, 1.0}.

Examples:
  # Evaluate a learned policy at the same reward it was trained on:
  python scripts/eval_constant_baselines.py \\
      --policy runs/ppo/ppo_surrogate_constant_inflow_20260512_001054/best_model.zip \\
      --seed 0

  # Evaluate the same policy at a different beta:
  python scripts/eval_constant_baselines.py \\
      --policy runs/ppo/.../best_model.zip --seed 0 --beta 1.0

  # Evaluate a constant baseline at a specific beta:
  python scripts/eval_constant_baselines.py --policy u=1.0 --seed 0 --beta 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl.surrogate_env import SurrogateEnv, find_latest_checkpoint  # noqa: E402


def _resolve_env_config(policy_arg: str, overrides: dict) -> dict:
    """Resolve the env config from either a saved PPO run dir or defaults."""
    if policy_arg.startswith("u="):
        # Constant-policy mode — build a minimal env config from the
        # default surrogate YAML, overriding reward weights.
        default_yaml = PROJECT_ROOT / "configs" / "rl" / "ppo_surrogate.yaml"
        with default_yaml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        env_cfg = dict(cfg["env"])
    else:
        # Learned-policy mode — read the config snapshot the run was
        # trained against, so the env matches training exactly.
        policy_path = Path(policy_arg)
        run_dir = policy_path.parent
        snapshot = run_dir / "config.yaml"
        if not snapshot.exists():
            raise FileNotFoundError(
                f"No config.yaml next to {policy_path}; cannot reconstruct env."
            )
        with snapshot.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        env_cfg = dict(cfg["env"])

    # Resolve "auto" checkpoint reference
    ckpt = env_cfg.get("surrogate_checkpoint")
    if ckpt in (None, "auto", "latest"):
        latest = find_latest_checkpoint(PROJECT_ROOT / "runs" / "surrogate")
        if latest is None:
            raise FileNotFoundError("No surrogate checkpoint under runs/surrogate/.")
        env_cfg["surrogate_checkpoint"] = str(latest)

    # Resolve sumo_config to absolute path so SurrogateEnv finds it
    sc = env_cfg.get("sumo_config")
    if sc and not Path(sc).is_absolute():
        env_cfg["sumo_config"] = str(PROJECT_ROOT / sc)

    # Apply reward-weight overrides
    if overrides:
        reward_cfg = dict(env_cfg.get("reward") or {})
        for k, v in overrides.items():
            if v is not None:
                reward_cfg[k] = float(v)
        env_cfg["reward"] = reward_cfg

    return env_cfg


def _make_action_callback(policy_arg: str):
    """Return a function obs -> action for the given policy spec."""
    if policy_arg.startswith("u="):
        u = float(policy_arg.removeprefix("u="))
        const_action = np.array([u], dtype=np.float32)

        def _const(_obs):
            return const_action

        return _const, f"constant u={u:.2f}"

    # Learned-policy mode
    from stable_baselines3 import PPO

    model = PPO.load(str(policy_arg))

    def _learned(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _learned, f"learned ({Path(policy_arg).parent.name})"


def rollout_policy(
    policy_arg: str,
    seed: int,
    beta: float | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
) -> dict:
    """Run a deterministic 120-step rollout and return summary stats."""
    env_cfg = _resolve_env_config(
        policy_arg,
        overrides={"alpha": alpha, "beta": beta, "gamma": gamma},
    )
    env = SurrogateEnv(
        surrogate_checkpoint=env_cfg["surrogate_checkpoint"],
        config=env_cfg,
    )
    action_fn, label = _make_action_callback(policy_arg)

    obs, _ = env.reset(seed=seed)
    actions: list[float] = []
    densities: list[np.ndarray] = []
    rewards: list[float] = []
    queues: list[float] = []
    for _ in range(env.T_ctrl):
        action = action_fn(obs)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, _, info = env.step(action)
        actions.append(float(action[0]))
        densities.append(info["density_phys"])
        rewards.append(float(reward))
        queues.append(float(info["analytical_queue"]))
        if terminated:
            break
    densities_arr = np.stack(densities)
    return {
        "label": label,
        "seed": seed,
        "reward_weights": dict(env_cfg["reward"]),
        "total_reward": float(sum(rewards)),
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
        "density_mean": float(densities_arr.mean()),
        "density_std": float(densities_arr.std()),
        "density_min": float(densities_arr.min()),
        "density_max": float(densities_arr.max()),
        "queue_final": float(queues[-1]),
        "queue_max": float(max(queues)),
        "n_steps": len(rewards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned or constant policy on SurrogateEnv")
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to a PPO best_model.zip, or a constant spec like u=0.5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=None, help="Override reward.beta")
    parser.add_argument("--alpha", type=float, default=None, help="Override reward.alpha")
    parser.add_argument("--gamma", type=float, default=None, help="Override reward.gamma")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as one JSON line (for sweep drivers)",
    )
    args = parser.parse_args()

    result = rollout_policy(
        policy_arg=args.policy,
        seed=args.seed,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
    )

    if args.json:
        print(json.dumps(result))
    else:
        print(f"== {result['label']} (seed={result['seed']}, weights={result['reward_weights']}) ==")
        print(f"  total_reward = {result['total_reward']:.2f}")
        print(f"  action       mean={result['action_mean']:.3f} std={result['action_std']:.3f} "
              f"min={result['action_min']:.3f} max={result['action_max']:.3f}")
        print(f"  density      mean={result['density_mean']:.2f} std={result['density_std']:.2f} "
              f"min={result['density_min']:.2f} max={result['density_max']:.2f}")
        print(f"  queue        final={result['queue_final']:.1f} max={result['queue_max']:.1f}")


if __name__ == "__main__":
    main()
