"""
SUMO-side counterpart of scripts/eval_constant_baselines.py.

Evaluates a learned PPO policy OR a constant ramp-metering policy
(`u=0.0`, `u=0.5`, `u=1.0`, etc.) inside SumoEnv (live TraCI SUMO),
with configurable reward weights.

Use cases:
- M6.4 — native eval of a SUMO-trained PPO policy.
- M6.5 — transfer eval of a surrogate-trained PPO policy in SUMO.
- M6.6 / M7 — constant baselines in SumoEnv, and the constant-u sweep that
  feeds scripts/balance_reward_terms.py (per-step term arrays are included
  in the `--json` output for that purpose).

Wall-clock note: each SUMO episode takes ~12–15 seconds on the M6 machine
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

  # transfer: surrogate-trained policy rolled out in SUMO:
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

REWARD_OVERRIDE_KEYS = ("delta", "beta", "gamma", "q_ref", "queue_norm", "sigma_ref")


def _load_sumo_env_config(overrides: dict, config_path: Path | None = None) -> dict:
    """Read a PPO config's env block (default configs/rl/ppo_sumo.yaml) and apply overrides."""
    path = config_path or (PROJECT_ROOT / "configs" / "rl" / "ppo_sumo.yaml")
    with path.open("r", encoding="utf-8") as f:
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


def apply_sumo_overrides(env_cfg: dict, specs: list[str] | None) -> dict:
    """Apply `section.key=value` specs to env_cfg["sumo_overrides"].

    Values are parsed with yaml.safe_load (so `5` -> int, `desired` -> str);
    `simulation.sumo_extra_args` accepts a whitespace-separated list, e.g.
    `simulation.sumo_extra_args=--extrapolate-departpos`.
    """
    for spec in specs or []:
        key, _, raw = spec.partition("=")
        if not _ or "." not in key:
            raise ValueError(f"--sumo-override expects section.key=value, got {spec!r}")
        section, name = key.split(".", 1)
        value = raw.split() if name == "sumo_extra_args" else yaml.safe_load(raw)
        env_cfg.setdefault("sumo_overrides", {}).setdefault(section, {})[name] = value
    return env_cfg


def _make_action_callback(policy_arg: str, env_cfg: dict | None = None):
    """Return a callable obs -> action (with optional .reset()) for the policy spec.

    Specs: `u=0.5` (constant), `alinea:ki=35,rho=30,det=14` /
    `pialinea:kp=4,ki=35,rho=30,det=14` (feedback baselines, see
    rl.baseline_controllers), or a path to a PPO .zip.
    """
    if policy_arg.startswith("u="):
        u = float(policy_arg.removeprefix("u="))
        const_action = np.array([u], dtype=np.float32)

        def _const(_obs):
            return const_action

        return _const, f"constant u={u:.2f}"

    from rl.baseline_controllers import is_controller_spec, make_controller

    if is_controller_spec(policy_arg):
        controller = make_controller(policy_arg, env_cfg or {})
        return controller, controller.label

    from stable_baselines3 import PPO

    model = PPO.load(str(policy_arg))
    # Models trained with env.symmetric_action=true act in [-1, 1]; map back
    # to the env's metering rate u in [0, 1] (see train_ppo._make_env).
    symmetric = bool(np.asarray(model.action_space.low).reshape(-1)[0] < 0.0)

    def _learned(obs):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if symmetric:
            action = (action + 1.0) / 2.0
        return np.clip(action, 0.0, 1.0)

    label = f"learned ({Path(policy_arg).parent.name})"
    if symmetric:
        label += " [symmetric-action]"
    return _learned, label


def rollout_policy_sumo(
    policy_arg: str,
    seed: int,
    n_episodes: int = 1,
    delta: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    q_ref: float | None = None,
    queue_norm: float | None = None,
    sigma_ref: float | None = None,
    config_path: Path | None = None,
    env: SumoEnv | None = None,
    sumo_overrides: list[str] | None = None,
    network_dir: str | None = None,
    reset_options: dict | None = None,
) -> dict:
    """Roll a policy through SumoEnv for n_episodes; return summary stats.

    Per-episode summaries include the per-step arrays `outflow_vph_steps`,
    `queue_steps`, `std_steps`, `mean_density_steps`, `reward_steps` so that
    scripts/balance_reward_terms.py can re-weight the three reward terms
    offline without re-running SUMO.

    Pass an existing `env` to reuse one SumoEnv across several calls (the
    u-sweep does this); it is not closed in that case.
    """
    env_cfg = _load_sumo_env_config(
        overrides={
            "delta": delta,
            "beta": beta,
            "gamma": gamma,
            "q_ref": q_ref,
            "queue_norm": queue_norm,
            "sigma_ref": sigma_ref,
        },
        config_path=config_path,
    )
    apply_sumo_overrides(env_cfg, sumo_overrides)
    if network_dir:
        env_cfg["network_dir"] = network_dir
    own_env = env is None
    if own_env:
        env = SumoEnv(env_cfg)
    action_fn, label = _make_action_callback(policy_arg, env_cfg)

    ep_summaries: list[dict] = []
    try:
        for ep in range(n_episodes):
            obs, _info = env.reset(seed=seed + ep, options=reset_options)
            if hasattr(action_fn, "reset"):
                action_fn.reset()  # stateful feedback baselines start fresh per episode
            actions: list[float] = []
            densities: list[np.ndarray] = []
            rewards: list[float] = []
            queues: list[float] = []
            outflows: list[float] = []
            stds: list[float] = []
            means: list[float] = []
            term_sums = {"outflow_penalty": 0.0, "queue_penalty": 0.0, "std_penalty": 0.0}
            last_info: dict = {}
            for _ in range(env.T_ctrl):
                action = action_fn(obs)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                obs, reward, terminated, _, last_info = env.step(action)
                actions.append(float(action[0]))
                densities.append(np.asarray(last_info["density"]))
                rewards.append(float(reward))
                queues.append(float(last_info["analytical_queue"]))
                outflows.append(float(last_info["outflow_vph"]))
                stds.append(float(last_info["std_density"]))
                means.append(float(last_info["mean_density"]))
                # Penalties are magnitudes; they are masked to 0 while reward
                # warmup is active so the sums stay consistent with `reward`.
                if not last_info.get("reward_warmup_active", 0.0):
                    for key in term_sums:
                        term_sums[key] += float(last_info[key])
                if terminated:
                    break
            densities_arr = np.stack(densities)
            ep_summaries.append({
                "episode": ep,
                "seed_used": seed + ep,
                "total_reward": float(sum(rewards)),
                "outflow_penalty_sum": term_sums["outflow_penalty"],
                "queue_penalty_sum": term_sums["queue_penalty"],
                "std_penalty_sum": term_sums["std_penalty"],
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
                "outflow_vph_mean": float(np.mean(outflows)),
                "outflow_vph_min": float(np.min(outflows)),
                "outflow_vph_max": float(np.max(outflows)),
                "throughput_vph": float(last_info.get("throughput_vph", 0.0)),
                "teleports": int(last_info.get("teleports", 0)),
                "insert_rejected": int(last_info.get("insert_rejected", 0)),
                "pending_mainline_max": int(last_info.get("episode_pending_mainline_max", 0)),
                "pending_mainline_final": int(last_info.get("pending_mainline", 0)),
                "pending_ramp_max": int(last_info.get("episode_pending_ramp_max", 0)),
                "discarded_mainline": int(last_info.get("discarded_mainline", 0)),
                "discarded_ramp": int(last_info.get("discarded_ramp", 0)),
                "max_depart_delay_s": float(last_info.get("max_depart_delay_s", -1.0)),
                "n_steps": len(rewards),
                "demand_vph": float(last_info.get("demand_vph", float("nan"))),
                "ramp_demand_vph": float(last_info.get("ramp_demand_vph", float("nan"))),
                "ramp_discharge_vph": float(last_info.get("ramp_discharge_vph", float("nan"))),
                "reward_warmup_s": float(last_info.get("reward_warmup_s", 0.0)),
                "outflow_vph_steps": [float(v) for v in outflows],
                "queue_steps": [float(v) for v in queues],
                "std_steps": [float(v) for v in stds],
                "mean_density_steps": [float(v) for v in means],
                "reward_steps": [float(v) for v in rewards],
                "action_steps": [float(v) for v in actions],
            })
    finally:
        if own_env:
            env.close()

    if n_episodes == 1:
        summary = dict(ep_summaries[0])
        summary["label"] = label
        summary["policy"] = policy_arg
        summary["reward_weights"] = env_cfg.get("reward", {})
        return summary

    # Aggregate across episodes.
    total_rewards = [e["total_reward"] for e in ep_summaries]
    action_means = [e["action_mean"] for e in ep_summaries]
    return {
        "label": label,
        "policy": policy_arg,
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


def print_single_episode(result: dict, seed: int, n_episodes: int) -> None:
    weights = result.get("reward_weights", {})
    print(f"== {result['label']} (seed={seed}, n_episodes={n_episodes}, weights={weights}) ==")
    print(f"  total_reward = {result['total_reward']:.2f}  "
          f"[outflow {-result['outflow_penalty_sum']:.1f} | "
          f"queue {-result['queue_penalty_sum']:.1f} | "
          f"std {-result['std_penalty_sum']:.1f}]")
    print(f"  action       mean={result['action_mean']:.3f} std={result['action_std']:.3f} "
          f"min={result['action_min']:.3f} max={result['action_max']:.3f}")
    print(f"  density      mean={result['density_mean']:.2f} std={result['density_std']:.2f} "
          f"min={result['density_min']:.2f} max={result['density_max']:.2f}")
    print(f"  queue        final={result['queue_final']:.1f} max={result['queue_max']:.1f}")
    print(f"  outflow      mean={result['outflow_vph_mean']:.0f} vph "
          f"(min {result['outflow_vph_min']:.0f}, max {result['outflow_vph_max']:.0f}); "
          f"throughput {result['throughput_vph']:.0f} vph; "
          f"teleports={result['teleports']}, rejected={result['insert_rejected']}")
    print(f"  insertion    pending mainline max={result['pending_mainline_max']} "
          f"final={result['pending_mainline_final']}; pending ramp max={result['pending_ramp_max']}; "
          f"discarded mainline={result['discarded_mainline']} ramp={result['discarded_ramp']} "
          f"(max_depart_delay_s={result['max_depart_delay_s']:g})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy in live SUMO via SumoEnv")
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to a PPO best_model.zip, or a constant spec like u=0.5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="PPO config whose env block to use (default configs/rl/ppo_sumo.yaml)",
    )
    parser.add_argument("--delta", type=float, default=None, help="Override reward.delta")
    parser.add_argument("--beta", type=float, default=None, help="Override reward.beta")
    parser.add_argument("--gamma", type=float, default=None, help="Override reward.gamma")
    parser.add_argument("--q-ref", type=float, default=None, help="Override reward.q_ref (veh/h)")
    parser.add_argument("--queue-norm", type=float, default=None, help="Override reward.queue_norm")
    parser.add_argument("--sigma-ref", type=float, default=None, help="Override reward.sigma_ref")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as one JSON line (for sweep drivers)",
    )
    parser.add_argument(
        "--sumo-override",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="SUMO scenario override, e.g. vehicle.depart_speed=desired, "
             "simulation.max_depart_delay_s=5, simulation.sumo_extra_args=--extrapolate-departpos",
    )
    parser.add_argument("--network-dir", default=None,
                        help="separate network dir (use with --sumo-override so training files are untouched)")
    args = parser.parse_args()

    result = rollout_policy_sumo(
        policy_arg=args.policy,
        seed=args.seed,
        n_episodes=args.n_episodes,
        delta=args.delta,
        beta=args.beta,
        gamma=args.gamma,
        q_ref=args.q_ref,
        queue_norm=args.queue_norm,
        sigma_ref=args.sigma_ref,
        config_path=args.config,
        sumo_overrides=args.sumo_override,
        network_dir=args.network_dir,
    )

    if args.json:
        print(json.dumps(result))
    elif args.n_episodes == 1:
        print_single_episode(result, args.seed, args.n_episodes)
    else:
        weights = result.get("reward_weights", {})
        print(f"== {result['label']} (seed={args.seed}, n_episodes={args.n_episodes}, weights={weights}) ==")
        print(f"  total_reward    mean={result['total_reward_mean']:.2f} std={result['total_reward_std']:.2f}")
        print(f"  range           [{result['total_reward_min']:.2f}, {result['total_reward_max']:.2f}]")
        print(f"  action mean     {result['action_mean']:.3f} (std across eps {result['action_std_across_episodes']:.3f})")


if __name__ == "__main__":
    main()
