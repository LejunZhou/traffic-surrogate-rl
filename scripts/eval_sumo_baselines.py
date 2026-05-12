"""
SUMO-side counterpart of scripts/eval_constant_baselines.py.

Evaluates a learned PPO policy OR a constant ramp-metering policy
(`u=0.0`, `u=0.5`, `u=1.0`, etc.) inside SumoEnv (live TraCI SUMO),
with configurable reward weights.

Use cases (Milestone 6):
- 6.4 — native eval of the M6 (SUMO-trained) PPO policy.
- 6.5 — transfer eval of the M5c (surrogate-trained) PPO policy in SUMO.
- 6.6 — constant baselines u=0/0.5/1.0 in SumoEnv for comparison.

Wall-clock note: each SUMO episode takes ~12–15 seconds on this machine
(see milestone_6_progress.md §6.1). Constant policies are deterministic
so one episode per policy is enough; learned policies should run with
`--n-episodes 3` for a noise estimate.

Examples:
  # constant baseline:
  python scripts/eval_sumo_baselines.py --policy u=0.5 --seed 0

  # learned policy (auto-uses the project's ppo_sumo.yaml env config):
  python scripts/eval_sumo_baselines.py \\
      --policy runs/rl/ppo_sumo_constant_inflow_m6_seed0_.../best_model.zip \\
      --seed 0

  # transfer: M5c surrogate-trained policy rolled out in SUMO:
  python scripts/eval_sumo_baselines.py \\
      --policy runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_.../best_model.zip \\
      --seed 0
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

from rl.sumo_env_wrapper import SumoEnv  # noqa: E402


def _load_sumo_env_config(overrides: dict) -> dict:
    """Read configs/rl/ppo_sumo.yaml's env block and apply weight overrides."""
    with (PROJECT_ROOT / "configs" / "rl" / "ppo_sumo.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"])
    env_cfg["project_root"] = str(PROJECT_ROOT)

    if overrides:
        reward_cfg = dict(env_cfg.get("reward") or {})
        for k, v in overrides.items():
            if v is not None:
                reward_cfg[k] = float(v)
        env_cfg["reward"] = reward_cfg

    # Resolve sumo_config to absolute if relative.
    sc = env_cfg.get("sumo_config")
    if sc and not Path(sc).is_absolute():
        env_cfg["sumo_config"] = str(PROJECT_ROOT / sc)
    return env_cfg


def _make_action_callback(policy_arg: str):
    """Return a function obs -> action for the given policy spec."""
    if policy_arg.startswith("u="):
        u = float(policy_arg.removeprefix("u="))
        const_action = np.array([u], dtype=np.float32)

        def _const(_obs):
            return const_action

        return _const, f"constant u={u:.2f}"

    from stable_baselines3 import PPO

    model = PPO.load(str(policy_arg))

    def _learned(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _learned, f"learned ({Path(policy_arg).parent.name})"


def rollout_policy_sumo(
    policy_arg: str,
    seed: int,
    n_episodes: int = 1,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    rho_freeflow: float | None = None,
    queue_norm: float | None = None,
) -> dict:
    """Roll a policy through SumoEnv for n_episodes; return summary stats."""
    env_cfg = _load_sumo_env_config(
        overrides={
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "rho_freeflow": rho_freeflow,
            "queue_norm": queue_norm,
        }
    )
    env = SumoEnv(env_cfg)
    action_fn, label = _make_action_callback(policy_arg)

    ep_summaries: list[dict] = []
    try:
        for ep in range(n_episodes):
            obs, _info = env.reset(seed=seed + ep)
            actions: list[float] = []
            densities: list[np.ndarray] = []
            rewards: list[float] = []
            queues: list[float] = []
            last_info: dict = {}
            for _ in range(env.T_ctrl):
                action = action_fn(obs)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                obs, reward, terminated, _, last_info = env.step(action)
                actions.append(float(action[0]))
                densities.append(np.asarray(last_info["density"]))
                rewards.append(float(reward))
                queues.append(float(last_info["analytical_queue"]))
                if terminated:
                    break
            densities_arr = np.stack(densities)
            ep_summaries.append({
                "episode": ep,
                "seed_used": seed + ep,
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
                "throughput_vph": float(last_info.get("throughput_vph", 0.0)),
                "teleports": int(last_info.get("teleports", 0)),
                "n_steps": len(rewards),
            })
    finally:
        env.close()

    if n_episodes == 1:
        summary = dict(ep_summaries[0])
        summary["label"] = label
        summary["reward_weights"] = env_cfg.get("reward", {})
        return summary

    # Aggregate across episodes.
    total_rewards = [e["total_reward"] for e in ep_summaries]
    action_means = [e["action_mean"] for e in ep_summaries]
    return {
        "label": label,
        "n_episodes": n_episodes,
        "reward_weights": env_cfg.get("reward", {}),
        "total_reward_mean": float(np.mean(total_rewards)),
        "total_reward_std": float(np.std(total_rewards)),
        "total_reward_min": float(np.min(total_rewards)),
        "total_reward_max": float(np.max(total_rewards)),
        "action_mean": float(np.mean(action_means)),
        "action_std_across_episodes": float(np.std(action_means)),
        "episodes": ep_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy in live SUMO via SumoEnv")
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to a PPO best_model.zip, or a constant spec like u=0.5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rho-freeflow", type=float, default=None)
    parser.add_argument("--queue-norm", type=float, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as one JSON line (for sweep drivers)",
    )
    args = parser.parse_args()

    result = rollout_policy_sumo(
        policy_arg=args.policy,
        seed=args.seed,
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        rho_freeflow=args.rho_freeflow,
        queue_norm=args.queue_norm,
    )

    if args.json:
        print(json.dumps(result))
    else:
        # Single-episode print mirrors eval_constant_baselines.py.
        weights = result.get("reward_weights", {})
        print(f"== {result['label']} (seed={args.seed}, n_episodes={args.n_episodes}, weights={weights}) ==")
        if args.n_episodes == 1:
            print(f"  total_reward = {result['total_reward']:.2f}")
            print(f"  action       mean={result['action_mean']:.3f} std={result['action_std']:.3f} "
                  f"min={result['action_min']:.3f} max={result['action_max']:.3f}")
            print(f"  density      mean={result['density_mean']:.2f} std={result['density_std']:.2f} "
                  f"min={result['density_min']:.2f} max={result['density_max']:.2f}")
            print(f"  queue        final={result['queue_final']:.1f} max={result['queue_max']:.1f}")
            print(f"  throughput   {result['throughput_vph']:.0f} vph, teleports={result['teleports']}")
        else:
            print(f"  total_reward    mean={result['total_reward_mean']:.2f} std={result['total_reward_std']:.2f}")
            print(f"  range           [{result['total_reward_min']:.2f}, {result['total_reward_max']:.2f}]")
            print(f"  action mean     {result['action_mean']:.3f} (std across eps {result['action_std_across_episodes']:.3f})")


if __name__ == "__main__":
    main()
