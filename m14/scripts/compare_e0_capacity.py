"""
Compare the merge capacity of two E0 screening reports (e.g. the 60 km/h ramp against the
120 km/h reference in configs/reference/e0_ramp120kmh.json).

Capacity proxy per profile: the lowest constant meter rate u at which the merge breaks down
(`breakdown_u_first`; grid step 0.1 = 0.1 * D veh/h). A lower first-breakdown rate on the
same demand profile means a lower merge capacity. Verdict on the mean shift over the
profiles that break down in both reports, in grid steps (120 veh/h at D = 1200):
  < 0.5  keep the capacity-tied constants (storage-mandatory threshold, feedforward
         capacity range, demand ceilings);
  0.5-1  keep them: the same size as the 30 deg -> 10 deg entry change (-67 veh/h, 5 of 9
         profiles one step lower), which the published study absorbed without rescaling;
  >= 1   rescale them before generating round-0 data.

Example: python scripts/compare_e0_capacity.py --new runs/study/m14/e0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compare(old: dict, new: dict) -> dict:
    D = float(new.get("meter_discharge_vph", 1200.0))
    rows = []
    for key, g_new in new["gate"].items():
        g_old = old["gate"].get(key)
        if g_old is None:
            continue
        if abs(float(g_old["peak_total"]) - float(g_new["peak_total"])) > 1.0:
            raise ValueError(f"profile {key}: peak totals differ ({g_old['peak_total']:.0f} vs {g_new['peak_total']:.0f}); "
                             "the reports use different E0 profiles")
        u_old, u_new = g_old.get("breakdown_u_first"), g_new.get("breakdown_u_first")
        shift = None if u_old is None or u_new is None else round((float(u_new) - float(u_old)) * D)
        rows.append({"profile": key, "peak_total": float(g_new["peak_total"]), "storage_needed": bool(g_new.get("storage_needed")),
                     "breakdown_u_first_old": u_old, "breakdown_u_first_new": u_new, "shift_vph": shift,
                     "best_constant_old": g_old.get("best_constant_u"), "best_constant_new": g_new.get("best_constant_u"),
                     "best_constant_return_old": g_old.get("best_constant_return"),
                     "best_constant_return_new": g_new.get("best_constant_return")})
    shifts = [r["shift_vph"] for r in rows if r["shift_vph"] is not None]
    mean_shift = float(np.mean(shifts)) if shifts else float("nan")
    step = 0.1 * D
    steps = abs(mean_shift) / step if shifts else 0.0
    rescale = steps >= 1.0
    if rescale:
        verdict = "capacity moved by >= one grid step on average: rescale the capacity-tied constants before round 0"
    elif steps >= 0.5:
        verdict = ("capacity moved by half to one grid step, like the 30 -> 10 deg change the published study absorbed "
                   "without rescaling: keep the constants, report the shift")
    else:
        verdict = "capacity within half a grid step of the reference: keep the constants"
    return {"meter_discharge_vph": D, "grid_step_vph": step, "rows": rows,
            "n_compared": len(shifts), "n_down": sum(s < 0 for s in shifts), "n_up": sum(s > 0 for s in shifts),
            "n_same": sum(s == 0 for s in shifts), "mean_shift_vph": mean_shift,
            "gate_old": old.get("gate_summary"), "gate_new": new.get("gate_summary"),
            "mean_shift_steps": steps, "rescale_recommended": rescale, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="runs/study/m14/e0.json")
    ap.add_argument("--old", default="configs/reference/e0_ramp120kmh.json")
    ap.add_argument("--out", default="runs/study/m14/e0_capacity_comparison.json")
    args = ap.parse_args()
    path = lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
    res = compare(json.loads(path(args.old).read_text(encoding="utf-8")), json.loads(path(args.new).read_text(encoding="utf-8")))
    path(args.out).parent.mkdir(parents=True, exist_ok=True)
    path(args.out).write_text(json.dumps(res, indent=1))
    print(f"{'profile':>7} {'peak':>6} {'storage':>7} {'first bd u old':>14} {'new':>5} {'shift veh/h':>11} "
          f"{'best u old':>10} {'new':>5}")
    for r in res["rows"]:
        print(f"{r['profile']:>7} {r['peak_total']:6.0f} {str(r['storage_needed']):>7} {str(r['breakdown_u_first_old']):>14} "
              f"{str(r['breakdown_u_first_new']):>5} {str(r['shift_vph']):>11} {str(r['best_constant_old']):>10} "
              f"{str(r['best_constant_new']):>5}")
    print(f"\n{res['n_compared']} profiles break down in both: {res['n_down']} lower, {res['n_same']} same, {res['n_up']} higher; "
          f"mean shift {res['mean_shift_vph']:+.0f} veh/h (grid step {res['grid_step_vph']:.0f})")
    for label in ("gate_old", "gate_new"):
        g = res[label] or {}
        print(f"E0 gate {label[5:]}: {g.get('n_storage_passed')}/{g.get('n_storage_needed')} storage-mandatory pass "
              f"(single constant u = {g.get('single_constant_u')})")
    print(f"\nVERDICT: {res['verdict']}\nwritten: {path(args.out)}")


if __name__ == "__main__":
    main()
