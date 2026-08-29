"""
Forced-breakdown recoverability test in SUMO (M7 §7.6).

Runs one SumoEnv episode with a two-phase constant policy: u = --u-jam for
the first --jam-minutes (gridlocks the merge at u >= 0.7), then u = --u-after
for the rest of the hour. Prints a 5-minute-window table of outflow, mean
density, pending / discarded mainline insertions and ramp queue, so the
post-jam SUMO insertion artifact (pending vehicles re-inserted at ~1550 vph
with the default --max-depart-delay -1) can be checked against the fix
(`simulation.max_depart_delay_s`).

Usage:
  python scripts/run_forced_jam_sumo.py --u-jam 1.0 --jam-minutes 10 --u-after 0.0
  python scripts/run_forced_jam_sumo.py ... --max-depart-delay -1     # SUMO default
  python scripts/run_forced_jam_sumo.py ... --max-depart-delay 5      # fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_sumo_baselines import _load_sumo_env_config, apply_sumo_overrides  # noqa: E402
from rl.sumo_env_wrapper import SumoEnv  # noqa: E402


def run(env: SumoEnv, u_jam: float, jam_s: float, u_after: float, seed: int,
        phases: list[tuple[float, float]] | None = None) -> dict:
    """phases: optional [(u, duration_s), ...] schedule overriding u_jam/jam_s/u_after."""
    def u_at(t: float) -> float:
        if phases:
            acc = 0.0
            for u, dur in phases:
                acc += dur
                if t < acc:
                    return u
            return phases[-1][0]
        return u_jam if t < jam_s else u_after

    obs, _ = env.reset(seed=seed)
    steps: list[dict] = []
    total = 0.0
    info: dict = {}
    for k in range(env.T_ctrl):
        u = u_at(k * env.dt_ctrl)
        obs, r, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        total += float(r)
        steps.append({
            "k": k, "t_s": (k + 1) * env.dt_ctrl, "u": u, "reward": float(r),
            "outflow_vph": float(info["outflow_vph"]),
            "mean_density": float(info["mean_density"]),
            "queue": float(info["analytical_queue"]),
            "pending_mainline": int(info["pending_mainline"]),
            "pending_ramp": int(info["pending_ramp"]),
            "discarded_mainline": int(info["discarded_mainline"]),
            "discarded_ramp": int(info["discarded_ramp"]),
            "ramp_released": int(info["ramp_released"]),
            "ramp_arrivals": int(info["ramp_arrivals"]),
            "entry_speed_kmh": float(np.asarray(info["speed"]).reshape(-1)[0]) if info.get("speed") is not None else float("nan"),
        })
        if terminated or truncated:
            break
    return {
        "u_jam": u_jam, "jam_s": jam_s, "u_after": u_after, "seed": seed, "phases": phases,
        "ramp_demand_vph": float(info.get("ramp_demand_vph", float("nan"))),
        "ramp_discharge_vph": float(info.get("ramp_discharge_vph", float("nan"))),
        "max_depart_delay_s": float(info.get("max_depart_delay_s", -1.0)),
        "total_reward": total,
        "arrived_vehicles": int(info["arrived_vehicles"]),
        "throughput_vph": float(info["throughput_vph"]),
        "teleports": int(info["teleports"]),
        "insert_rejected": int(info["insert_rejected"]),
        "pending_mainline_final": int(info["pending_mainline"]),
        "pending_mainline_max": int(info["episode_pending_mainline_max"]),
        "pending_ramp_max": int(info["episode_pending_ramp_max"]),
        "discarded_mainline": int(info["discarded_mainline"]),
        "discarded_ramp": int(info["discarded_ramp"]),
        "queue_final": float(info["analytical_queue"]),
        "steps": steps,
    }


def print_table(res: dict, window_min: float = 5.0) -> None:
    steps = res["steps"]
    dt = steps[1]["t_s"] - steps[0]["t_s"] if len(steps) > 1 else 30
    per_window = max(int(round(window_min * 60 / dt)), 1)
    sched = (" -> ".join(f"u={u:g} x{d/60:.0f}min" for u, d in res["phases"]) if res.get("phases")
             else f"u={res['u_jam']} for {res['jam_s']/60:.0f} min, then u={res['u_after']}")
    print(f"\n== forced jam: {sched} (seed {res['seed']}, max_depart_delay_s={res['max_depart_delay_s']:g}, "
          f"ramp {res.get('ramp_demand_vph', float('nan')):.0f} vph arrivals / {res.get('ramp_discharge_vph', float('nan')):.0f} vph discharge) ==")
    print(f"{'window':>12} {'u':>4} {'q_out':>6} {'rho':>6} {'v_entry':>7} {'pend_max':>8} {'disc_main':>9} "
          f"{'disc_ramp':>9} {'ramp_out':>8} {'queue_end':>9} {'reward':>8}")
    for i in range(0, len(steps), per_window):
        w = steps[i:i + per_window]
        print(f"{w[0]['t_s']-dt:5.0f}-{w[-1]['t_s']:5.0f}s {w[-1]['u']:4.1f} "
              f"{np.mean([s['outflow_vph'] for s in w]):6.0f} "
              f"{np.mean([s['mean_density'] for s in w]):6.1f} "
              f"{np.mean([s['entry_speed_kmh'] for s in w]):7.1f} "
              f"{max(s['pending_mainline'] for s in w):8d} "
              f"{w[-1]['discarded_mainline']:9d} {w[-1]['discarded_ramp']:9d} "
              f"{sum(s['ramp_released'] for s in w) * 3600 / (len(w) * dt):8.0f} "
              f"{w[-1]['queue']:9.0f} {sum(s['reward'] for s in w):8.1f}")
    print(f"  return={res['total_reward']:.1f}  arrived={res['arrived_vehicles']}  "
          f"throughput={res['throughput_vph']:.0f} vph  teleports={res['teleports']}  "
          f"pending mainline max={res['pending_mainline_max']} final={res['pending_mainline_final']}  "
          f"pending ramp max={res['pending_ramp_max']}  "
          f"discarded mainline={res['discarded_mainline']} ramp={res['discarded_ramp']}  "
          f"queue_final={res['queue_final']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-breakdown recoverability test in SumoEnv")
    parser.add_argument("--u-jam", type=float, default=1.0)
    parser.add_argument("--jam-minutes", type=float, default=10.0)
    parser.add_argument("--u-after", type=float, default=0.0)
    parser.add_argument("--phases", default=None,
                        help='schedule "u:seconds,u:seconds,..." overriding --u-jam/--jam-minutes/--u-after, '
                             'e.g. "1.0:600,0.0:300,1.0:2700" = jam, close, flush')
    parser.add_argument("--ramp-demand", type=float, default=None, help="ramp arrival rate for this episode (vph)")
    parser.add_argument("--demand", type=float, default=None,
                        help="mainline demand for this episode (vph); overrides the RL config's demand_levels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "rl" / "ppo_sumo.yaml")
    parser.add_argument("--max-depart-delay", type=float, default=None,
                        help="override simulation.max_depart_delay_s (-1 = SUMO default, wait forever)")
    parser.add_argument("--speed-dev", type=float, default=None, help="override vehicle.speed_dev")
    parser.add_argument("--depart-speed", default=None,
                        help="override vehicle.depart_speed for the mainline flow (max|desired|speedLimit|...)")
    parser.add_argument("--sumo-override", action="append", default=None, metavar="SECTION.KEY=VALUE",
                        help="generic scenario override (see eval_sumo_baselines.py)")
    parser.add_argument("--network-dir", default="data/raw/rl_network_forced_jam")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    env_cfg = _load_sumo_env_config(overrides={}, config_path=args.config)
    env_cfg["network_dir"] = args.network_dir
    overrides: dict = {}
    if args.max_depart_delay is not None:
        overrides.setdefault("simulation", {})["max_depart_delay_s"] = float(args.max_depart_delay)
    if args.speed_dev is not None:
        overrides.setdefault("vehicle", {})["speed_dev"] = float(args.speed_dev)
    if args.depart_speed is not None:
        overrides.setdefault("vehicle", {})["depart_speed"] = str(args.depart_speed)
    if overrides:
        env_cfg["sumo_overrides"] = overrides
    apply_sumo_overrides(env_cfg, args.sumo_override)

    phases = None
    if args.phases:
        phases = [(float(u), float(d)) for u, d in (p.split(":") for p in args.phases.split(","))]
    if args.ramp_demand is not None:
        env_cfg["ramp_demand_levels"] = [float(args.ramp_demand)]
    if args.demand is not None:
        env_cfg["demand_levels"] = [float(args.demand)]
    env = SumoEnv(env_cfg)
    try:
        res = run(env, args.u_jam, args.jam_minutes * 60.0, args.u_after, args.seed, phases=phases)
    finally:
        env.close()
    print_table(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
