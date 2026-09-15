"""
E0 — scenario characterisation for the time-varying demand family (M8, draft §4.4).

(i)   Per-block insertion check on 20 random training-family profiles at a
      pass-through metering rate: served vs offered vehicles, pending
      mainline insertions, jam flags.
(ii)  Capacity map: constant u in {0, 0.1, ..., 1.0} on 12 profiles;
      breakdown onset / recovery and return per (profile, u).
(iii) Store-and-flush schedules on the same 12 profiles: u_low in
      {0, 0.15, 0.3} during the mainline peak window, flush at u = 1 starting
      {0, 5, 10} min after the peak, pass-through (u = 0.6) before.

Gate: a dynamic schedule beats the best constant by >= 15 return units on
>= 2/3 of the peaked profiles. Every rollout is written to the round-0
store (reused as training data) and logged in the ledger (purpose dataset).

  PYTHONPATH=src python scripts/run_scenario_characterisation.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.behaviour_controllers import store_and_flush_schedule  # noqa: E402
from rl.reward import RewardWeights  # noqa: E402
from sumo_env.rollout import load_rollout_npz, rescore_return  # noqa: E402
from utils.config import load_config as _load_cfg  # noqa: E402
from sumo_env.demand_profiles import ProfileFamily  # noqa: E402
from sumo_env.parallel_rollouts import run_jobs  # noqa: E402
from sumo_env.rollout_store import RolloutStore  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.ledger import Ledger  # noqa: E402

CONST_GRID = [round(0.1 * i, 1) for i in range(11)]
# Schedule rates are defined in veh/h and converted to green fractions with the
# scenario's meter discharge D (u = rate / D, clipped to 1), so the same physical
# schedules are run on every scenario; for D = 1600 (v2/v3) the fractions are the
# original grid (u_pre 0.6, u_low 0/0.15/0.3, flush 1.0, u_low2 0.1/0.2/0.3,
# u_high2 0.5/0.7, ff u_min 0.05, insertion 0.65).
U_PRE_VPH = 960.0
U_LOW_VPH = [0.0, 240.0, 480.0]
FLUSH_LAG_MIN = [0.0, 5.0, 10.0]
U_HIGH_VPH = 1600.0
# second schedule family (added after the first E0 pass: flushing at u = 1
# after the peak jams the merge, so no store-and-flush beat a constant)
U_LOW2_VPH = [160.0, 320.0, 480.0]
U_HIGH2_VPH = [800.0, 1120.0]
FF_CAPACITY = [2300.0, 2400.0, 2500.0]
FF_LAG = [0, 2]
FF_UMIN_VPH = 80.0
INSERTION_VPH = 1040.0


def _meter_discharge_vph(env_cfg: dict) -> float:
    """The scenario's meter saturation flow D (env config overrides the scenario file)."""
    if env_cfg.get("ramp_discharge_vph") is not None:
        return float(env_cfg["ramp_discharge_vph"])
    sc = _load_cfg(str(PROJECT_ROOT / env_cfg["sumo_config"]))
    return float(sc["demand"]["ramp_discharge_vph"])


def _frac(rate_vph: float, D: float) -> float:
    return float(min(max(rate_vph / D, 0.0), 1.0))
GATE_MARGIN = 15.0
STORAGE_NEEDED_VPH = 2500.0   # profiles whose peak total exceeds ~capacity (2480-2500): storage is mandatory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/round0_mixture.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-insertion", type=int, default=20)
    ap.add_argument("--n-profiles", type=int, default=12)
    ap.add_argument("--out", default="_progress/m8_e0_characterisation.json")
    ap.add_argument("--study", default="m8_e0")
    ap.add_argument("--no-store", action="store_true", help="do not keep the rollouts as dataset")
    ap.add_argument("--reward-config", default="configs/rl/ppo_common.yaml",
                    help="returns in the report are re-scored under this config's env.reward (any form)")
    args = ap.parse_args()

    cfg = load_config(str(PROJECT_ROOT / args.config))
    env_cfg = dict(cfg["env"]); env_cfg["project_root"] = str(PROJECT_ROOT)
    out_cfg = cfg["output"]
    store = None if args.no_store else RolloutStore(PROJECT_ROOT / out_cfg["store_dir"])
    out_dir = None if store is None else str(store.root)
    family = ProfileFamily.load(PROJECT_ROOT / cfg["env"]["profiles"]["family"])
    profiles = [family.sample_by_key("e0", i) for i in range(args.n_profiles)]
    ins_profiles = [family.sample_by_key("e0", 100 + i) for i in range(args.n_insertion)]

    D = _meter_discharge_vph(env_cfg)
    U_PRE = _frac(U_PRE_VPH, D); U_HIGH = _frac(U_HIGH_VPH, D); FF_UMIN = _frac(FF_UMIN_VPH, D)
    U_LOW = [_frac(v, D) for v in U_LOW_VPH]; U_LOW2 = [_frac(v, D) for v in U_LOW2_VPH]; U_HIGH2 = [_frac(v, D) for v in U_HIGH2_VPH]
    print(f"[E0] meter discharge D = {D:.0f} veh/h: u_pre {U_PRE:.2f}, u_low {[round(u, 2) for u in U_LOW]}, flush {U_HIGH:.2f}, "
          f"u_low2 {[round(u, 2) for u in U_LOW2]}, u_high2 {[round(u, 2) for u in U_HIGH2]}, ff u_min {FF_UMIN:.3f}, insertion {_frac(INSERTION_VPH, D):.2f}")

    jobs = []
    # (i) insertion check at a pass-through rate (1040 vph >= any r_k)
    for i, p in enumerate(ins_profiles):
        jobs.append({"name": f"e0_insertion_{i:03d}", "profile": p.to_dict(), "sumo_seed": 60000 + i,
                     "controller": {"type": "constant", "u": _frac(INSERTION_VPH, D)}, "round": 0, "purpose": "dataset",
                     "e0": "insertion", "profile_draw": 100 + i, "out_dir": out_dir})
    # (ii) capacity map
    for i, p in enumerate(profiles):
        for u in CONST_GRID:
            jobs.append({"name": f"e0_constant_{i:02d}_u{int(round(u * 100)):03d}", "profile": p.to_dict(),
                         "sumo_seed": 61000 + i, "controller": {"type": "constant", "u": u}, "round": 0,
                         "purpose": "dataset", "e0": "capacity", "profile_draw": i, "out_dir": out_dir})
    # (iii) store-and-flush
    for i, p in enumerate(profiles):
        for u_low in U_LOW:
            for lag in FLUSH_LAG_MIN:
                spec = {"type": "store_flush", "u_pre": U_PRE, "u_low": u_low, "u_high": U_HIGH, "lead_min": 0.0, "lag_min": lag}
                jobs.append({"name": f"e0_flush_{i:02d}_low{int(round(u_low * 100)):03d}_lag{int(lag):02d}",
                             "profile": p.to_dict(), "sumo_seed": 61000 + i, "controller": spec, "round": 0,
                             "purpose": "dataset", "e0": "flush", "profile_draw": i, "out_dir": out_dir})
    # (iii b) moderate-flush store-and-flush + capacity-tracking feedforward
    for i, p in enumerate(profiles):
        for u_low in U_LOW2:
            for u_high in U_HIGH2:
                spec = {"type": "store_flush", "u_pre": U_PRE, "u_low": u_low, "u_high": u_high, "lead_min": 0.0, "lag_min": 0.0}
                jobs.append({"name": f"e0_flush2_{i:02d}_low{int(round(u_low * 100)):03d}_high{int(round(u_high * 100)):03d}",
                             "profile": p.to_dict(), "sumo_seed": 61000 + i, "controller": spec, "round": 0,
                             "purpose": "dataset", "e0": "flush2", "profile_draw": i, "out_dir": out_dir})
        for C in FF_CAPACITY:
            for lag in FF_LAG:
                spec = {"type": "feedforward", "capacity_vph": C, "lag_steps": lag, "u_min": FF_UMIN}
                jobs.append({"name": f"e0_ff_{i:02d}_C{int(C)}_lag{lag}", "profile": p.to_dict(), "sumo_seed": 61000 + i,
                             "controller": spec, "round": 0, "purpose": "dataset", "e0": "feedforward",
                             "profile_draw": i, "out_dir": out_dir})
    if store is not None:
        existing = {e["file"] for e in store.entries}
        jobs = [j for j in jobs if f"{j['name']}.npz" not in existing]
    print(f"[E0] {len(jobs)} rollouts to run ({args.n_insertion} insertion, {len(profiles)}x{len(CONST_GRID)} constant, "
          f"{len(profiles)}x{len(U_LOW) * len(FLUSH_LAG_MIN) + len(U_LOW2) * len(U_HIGH2) + len(FF_CAPACITY) * len(FF_LAG)} "
          f"dynamic schedules; existing files reused) on {args.workers} workers")
    t0 = time.time()
    results = run_jobs(jobs, env_cfg, workers=args.workers,
                       network_root=str(PROJECT_ROOT / out_cfg.get("network_dir", "data/plant_v2/network")))
    ledger = Ledger(args.study, PROJECT_ROOT)
    for r in results:
        if store is not None:
            store.add(r["path"], r["meta"])
        ledger.log(0, "dataset", "e0", r["meta"]["profile_draw"], r["meta"]["sumo_seed"],
                   r["meta"]["controller"]["type"], r["metrics"]["return"], r["metrics"]["breakdown"], r["metrics"]["wall_s"])
    if store is not None:
        store.save_index()
    # Metrics for the report: re-read every E0 rollout from the store and re-score
    # its return under the current training reward (any form), so the gate
    # reflects what PPO will optimise.
    rcfg = _load_cfg(str(PROJECT_ROOT / args.reward_config))["env"]["reward"]
    weights = RewardWeights.from_config(rcfg)
    metrics = {r["name"]: r["metrics"] for r in results}
    if store is not None:
        for e in store.entries:
            name = Path(e["file"]).stem
            if name.startswith("e0_"):
                arrays, meta = load_rollout_npz(store.root / e["file"])
                m = dict(meta["metrics"])
                m.update(rescore_return(arrays, weights, dt_ctrl_s=float(meta.get("dt_ctrl_s", 30.0)),
                                        warmup_s=float(rcfg.get("warmup_s", 90))))
                metrics[name] = m
    report_reward = {"config": args.reward_config, "form": weights.form}

    report = {"profiles": [p.to_dict() for p in profiles], "insertion": [], "capacity": {}, "flush": {}, "gate": {},
              "reward": report_reward, "meter_discharge_vph": D,
              "schedule_grid": {"u_pre": U_PRE, "u_low": U_LOW, "u_high": U_HIGH, "u_low2": U_LOW2, "u_high2": U_HIGH2, "ff_u_min": FF_UMIN}}
    # (i)
    for i, p in enumerate(ins_profiles):
        m = metrics[f"e0_insertion_{i:03d}"]
        report["insertion"].append({"profile": p.label, "peak_total": p.peak_total_vph, "served": m["served_veh"],
                                    "offered": m["offered_veh"], "served_frac": m["served_veh"] / max(m["offered_veh"], 1),
                                    "final_pending": m["final_pending_mainline"], "breakdown": m["breakdown"],
                                    "final_queue": m["final_queue"], "teleports": m["teleports"]})
    # (ii) / (iii)
    n_peaked = 0; n_gate = 0
    for i, p in enumerate(profiles):
        cap = {}
        for u in CONST_GRID:
            m = metrics[f"e0_constant_{i:02d}_u{int(round(u * 100)):03d}"]
            cap[f"{u:.1f}"] = {"return": m["return"], "breakdown": m["breakdown"], "onset_min": m["breakdown_onset_min"],
                               "recovered": m["recovered"], "final_queue": m["final_queue"], "tts": m["tts_veh_h"]}
        best_u = max(cap, key=lambda k: cap[k]["return"])
        fl = {}
        for u_low in U_LOW:
            for lag in FLUSH_LAG_MIN:
                m = metrics[f"e0_flush_{i:02d}_low{int(round(u_low * 100)):03d}_lag{int(lag):02d}"]
                fl[f"low{u_low:.2f}_lag{lag:.0f}"] = {"return": m["return"], "breakdown": m["breakdown"],
                                                       "final_queue": m["final_queue"], "tts": m["tts_veh_h"]}
        for u_low in U_LOW2:
            for u_high in U_HIGH2:
                m = metrics[f"e0_flush2_{i:02d}_low{int(round(u_low * 100)):03d}_high{int(round(u_high * 100)):03d}"]
                fl[f"low{u_low:.2f}_high{u_high:.2f}"] = {"return": m["return"], "breakdown": m["breakdown"],
                                                           "final_queue": m["final_queue"], "tts": m["tts_veh_h"]}
        for C in FF_CAPACITY:
            for lag in FF_LAG:
                m = metrics[f"e0_ff_{i:02d}_C{int(C)}_lag{lag}"]
                fl[f"ff_C{int(C)}_lag{lag}"] = {"return": m["return"], "breakdown": m["breakdown"],
                                                "final_queue": m["final_queue"], "tts": m["tts_veh_h"]}
        best_fl = max(fl, key=lambda k: fl[k]["return"])
        margin = fl[best_fl]["return"] - cap[best_u]["return"]
        peaked = p.is_peaked
        storage_needed = bool(p.peak_total_vph > STORAGE_NEEDED_VPH)
        passed = bool(peaked and margin >= GATE_MARGIN)
        n_peaked += int(peaked); n_gate += int(passed)
        n_storage = locals().get("n_storage", 0) + int(storage_needed)
        n_storage_pass = locals().get("n_storage_pass", 0) + int(storage_needed and margin >= GATE_MARGIN)
        report["capacity"][i] = cap
        report["flush"][i] = fl
        report["gate"][i] = {"profile": p.label, "peaked": peaked, "storage_needed": storage_needed, "peak_total": p.peak_total_vph,
                             "best_constant_u": best_u, "best_constant_return": cap[best_u]["return"],
                             "best_flush": best_fl, "best_flush_return": fl[best_fl]["return"],
                             "margin": margin, "passed": passed,
                             "breakdown_u_first": next((u for u in CONST_GRID if cap[f"{u:.1f}"]["breakdown"]), None)}
    frac = n_gate / max(n_peaked, 1)
    n_storage = sum(int(g["storage_needed"]) for g in report["gate"].values())
    n_storage_pass = sum(int(g["storage_needed"] and g["margin"] >= GATE_MARGIN) for g in report["gate"].values())
    frac_storage = n_storage_pass / max(n_storage, 1)
    report["gate_summary"] = {"n_peaked": n_peaked, "n_passed": n_gate, "fraction": frac,
                              "passed": bool(frac >= 2 / 3), "margin_required": GATE_MARGIN,
                              "n_storage_needed": n_storage, "n_storage_passed": n_storage_pass,
                              "fraction_storage_needed": frac_storage, "passed_storage_needed": bool(frac_storage >= 2 / 3),
                              "wall_s": time.time() - t0, "n_rollouts": len(results)}
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    print(f"\n== (i) insertion check (u = {_frac(INSERTION_VPH, D):.2f} = {INSERTION_VPH:.0f} veh/h pass-through) ==")
    for r in report["insertion"]:
        print(f"  {r['profile']:<40s} served/offered={r['served_frac']:.3f} pending_end={r['final_pending']:4.0f} "
              f"Qend={r['final_queue']:4.0f} bd={int(r['breakdown'])} tel={r['teleports']}")
    print(f"\n== (ii)+(iii) capacity map and dynamic-vs-constant margin (returns under {args.reward_config}: form={weights.form}) ==")
    print(f"  {'profile':<40s} {'peak':>5} {'best u':>6} {'R_const':>8} {'best schedule':>18} {'R_sched':>8} {'margin':>7} {'1st bd u':>8} gate")
    for i in range(len(profiles)):
        g = report["gate"][i]
        tag = "PASS" if g["passed"] else ("-" if not g["peaked"] else "fail")
        if g["storage_needed"]:
            tag += "*"
        print(f"  {g['profile']:<40s} {g['peak_total']:5.0f} {g['best_constant_u']:>6} {g['best_constant_return']:8.1f} "
              f"{g['best_flush']:>18} {g['best_flush_return']:8.1f} {g['margin']:7.1f} {str(g['breakdown_u_first']):>8} {tag}")
    gs = report["gate_summary"]
    print(f"\n  E0 gate (all peaked profiles): {gs['n_passed']}/{gs['n_peaked']} with margin >= {GATE_MARGIN}: "
          f"{'PASSED' if gs['passed'] else 'FAILED'}")
    print(f"  E0 gate (* = peak total > {STORAGE_NEEDED_VPH:.0f} vph, storage mandatory): "
          f"{gs['n_storage_passed']}/{gs['n_storage_needed']}: {'PASSED' if gs['passed_storage_needed'] else 'FAILED'}  "
          f"({gs['n_rollouts']} new rollouts, {gs['wall_s']:.0f} s)")
    print(f"  report: {out}")


if __name__ == "__main__":
    main()
