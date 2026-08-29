"""
Does the SUMO scenario (insertion settings, ramp model, detectors) behave
correctly across a range of mainline demands?

For every demand level and constant metering rate u it runs one episode and
checks that (a) insertion actually delivers the scheduled demand in free
flow (outflow ≈ demand + 800·u, pending insertions 0, free-flow entry speed),
(b) where the merge breaks down, and (c) that a forced breakdown (u = 1 for
10 min, then u = 0) still recovers to full insertion.

Usage:
  python scripts/check_demand_range_sumo.py --demands 1500 1600 1700 1800 1900 2000 \\
      --u 0.0 0.5 0.7 1.0 --config configs/rl/ppo_sumo_m7_run4.yaml \\
      --out _progress/m7_demand_range_check.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from eval_sumo_baselines import _load_sumo_env_config, apply_sumo_overrides  # noqa: E402
from rl.sumo_env_wrapper import SumoEnv  # noqa: E402

BREAKDOWN_DENSITY = 60.0


def run_episode(env: SumoEnv, demand: float, u_fn, seed: int, ramp_demand: float | None = None) -> dict:
    options = {"demand_vph": demand}
    if ramp_demand is not None:
        options["ramp_demand_vph"] = ramp_demand
    env.reset(seed=seed, options=options)
    rec = {"t": [], "u": [], "q_out": [], "rho": [], "v_entry": [], "pending": [], "reward": []}
    info: dict = {}
    for k in range(env.T_ctrl):
        u = float(u_fn(k * env.dt_ctrl))
        _, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        rec["t"].append((k + 1) * env.dt_ctrl); rec["u"].append(u); rec["reward"].append(float(r))
        rec["q_out"].append(float(info["outflow_vph"])); rec["rho"].append(float(info["mean_density"]))
        rec["v_entry"].append(float(np.asarray(info["speed"]).reshape(-1)[0]))
        rec["pending"].append(int(info["pending_mainline"]))
        if term or trunc:
            break
    t = np.array(rec["t"]); steady = t > 600  # skip fill-in transient
    return {
        "demand_vph": demand, "seed": seed,
        "ramp_demand_vph": float(info["ramp_demand_vph"]), "ramp_discharge_vph": float(info["ramp_discharge_vph"]),
        "total_reward": float(sum(rec["reward"])),
        "outflow_mean_steady": float(np.mean(np.array(rec["q_out"])[steady])),
        "entry_speed_steady_kmh": float(np.mean(np.array(rec["v_entry"])[steady])),
        "density_max": float(max(rec["rho"])),
        "breakdown": bool(max(rec["rho"]) > BREAKDOWN_DENSITY),
        "pending_max": int(max(rec["pending"])), "pending_final": int(rec["pending"][-1]),
        "discarded_mainline": int(info["discarded_mainline"]),
        "teleports": int(info["teleports"]), "insert_rejected": int(info["insert_rejected"]),
        "arrived": int(info["arrived_vehicles"]), "throughput_vph": float(info["throughput_vph"]),
        "queue_final": float(info["analytical_queue"]),
        "steps": rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario sanity check across mainline demand levels")
    ap.add_argument("--demands", type=float, nargs="+", default=[1500, 1600, 1700, 1800, 1900, 2000])
    ap.add_argument("--u", type=float, nargs="+", default=[0.0, 0.5, 0.7, 1.0])
    ap.add_argument("--ramp-demands", type=float, nargs="*", default=None,
                    help="ramp arrival rates to test (default: scenario ramp_demand_vph)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jam-minutes", type=float, default=10.0)
    ap.add_argument("--no-forced-jam", action="store_true")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "rl" / "ppo_sumo.yaml")
    ap.add_argument("--sumo-override", action="append", default=None, metavar="SECTION.KEY=VALUE")
    ap.add_argument("--network-dir", default="data/raw/rl_network_demand_check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env_cfg = _load_sumo_env_config(overrides={}, config_path=args.config)
    env_cfg["network_dir"] = args.network_dir
    env_cfg["demand_levels"] = [float(d) for d in args.demands]
    apply_sumo_overrides(env_cfg, args.sumo_override)
    ramp_levels = [float(r) for r in args.ramp_demands] if args.ramp_demands else None
    if ramp_levels:
        env_cfg["ramp_demand_levels"] = ramp_levels
    env = SumoEnv(env_cfg)
    discharge = env.ramp_discharge_vph
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        with args.out.open("w", encoding="utf-8") as f:
            print(f"{'demand':>6} {'ramp':>5} {'u':>4} {'expected':>8} {'q_out':>6} {'ratio':>6} {'v_entry':>7} {'rho_max':>7} {'pend_max':>8} {'disc':>4} {'tele':>4} {'queue':>6} {'return':>8}  note")
            for d in args.demands:
              for ramp in (ramp_levels or [env.ramp_demand_vph]):
                for u in args.u:
                    r = run_episode(env, d, lambda _t, u=u: u, args.seed, ramp_demand=ramp)
                    r["kind"] = "constant"; r["u"] = u
                    # steady ramp flow at constant u = min(green capacity, arrivals)
                    expected = d + min(discharge * u, ramp)
                    r["expected_outflow_vph"] = expected
                    r["delivery_ratio"] = r["outflow_mean_steady"] / expected
                    note = "BREAKDOWN" if r["breakdown"] else ("ok" if 0.97 <= r["delivery_ratio"] <= 1.03 else "SHORTFALL")
                    r["note"] = note
                    print(f"{d:6.0f} {ramp:5.0f} {u:4.2f} {expected:8.0f} {r['outflow_mean_steady']:6.0f} {r['delivery_ratio']:6.3f} "
                          f"{r['entry_speed_steady_kmh']:7.1f} {r['density_max']:7.0f} {r['pending_max']:8d} {r['discarded_mainline']:4d} "
                          f"{r['teleports']:4d} {r['queue_final']:6.0f} {r['total_reward']:8.1f}  {note}", flush=True)
                    f.write(json.dumps(r) + "\n"); f.flush(); rows.append(r)
                if not args.no_forced_jam:
                    jam_s = args.jam_minutes * 60.0
                    r = run_episode(env, d, lambda t, j=jam_s: 1.0 if t < j else 0.0, args.seed, ramp_demand=ramp)
                    r["kind"] = "forced_jam"; r["u"] = None
                    t = np.array(r["steps"]["t"]); post = t > 1200
                    r["post_jam_outflow_vph"] = float(np.mean(np.array(r["steps"]["q_out"])[post]))
                    r["post_jam_entry_speed_kmh"] = float(np.mean(np.array(r["steps"]["v_entry"])[post]))
                    r["post_jam_ratio"] = r["post_jam_outflow_vph"] / d
                    jammed = bool(max(np.array(r["steps"]["rho"])[t <= jam_s]) > BREAKDOWN_DENSITY)
                    r["jam_occurred"] = jammed
                    note = ("no jam at u=1" if not jammed else
                            ("recovered" if r["post_jam_ratio"] >= 0.99 and r["pending_final"] < r["pending_max"] else "NOT RECOVERED"))
                    r["note"] = note
                    print(f"{d:6.0f} {ramp:5.0f} {'jam':>4} {d:8.0f} {r['post_jam_outflow_vph']:6.0f} {r['post_jam_ratio']:6.3f} "
                          f"{r['post_jam_entry_speed_kmh']:7.1f} {r['density_max']:7.0f} {r['pending_max']:8d} {r['discarded_mainline']:4d} "
                          f"{r['teleports']:4d} {r['queue_final']:6.0f} {r['total_reward']:8.1f}  forced jam: {note} (pending final {r['pending_final']})", flush=True)
                    f.write(json.dumps(r) + "\n"); f.flush(); rows.append(r)
    finally:
        env.close()
    print(f"wrote {len(rows)} episodes to {args.out}")


if __name__ == "__main__":
    main()
