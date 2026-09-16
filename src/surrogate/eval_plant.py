"""
Validation metrics for the plant-model ensemble (M9, draft_pipeline.md §5.5).

Per held-out rollout the ensemble mean and spread of the density and exit-flow
fields are compared with the SUMO labels:

  * relative L2 of rho: global, free-flow cells (< 30 veh/km), shockwave band
    (30-60 upstream of the merge), jam cells (> 60)
  * relative L2 of the exit flow and the episode-sum outflow error
  * per-step (per-k) mean absolute density error
  * return-prediction error: the three-term return recomputed from the
    predicted fields with the stored queue / demand / q_ref versus from the
    labels, same reward weights
  * breakdown-onset error (min) and false / missed breakdown rates
  * ensemble calibration: |error| regressed on the ensemble std (slope, and
    the fraction of errors inside +-2 std)

Grouped per controller type (regime) and over everything. Used by
scripts/eval_surrogate_regimes.py and by the aggregation loop (spread on the
new on-policy rollouts before / after fine-tuning).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl.reward import RewardWeights
from sumo_env.rollout import breakdown_flags, load_rollout_npz, rescore_return
from surrogate.deeponet import DeepONetEnsemble

FREE_MAX, JAM_MIN = 30.0, 60.0


def _rel_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        if not mask.any():
            return float("nan")
        pred, true = pred[mask], true[mask]
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-6))


def episode_return(arrays: dict, density: np.ndarray, outflow: np.ndarray, weights: RewardWeights,
                   warmup_s: float, dt: float, dx_km: float) -> float:
    return rescore_return(arrays, weights, dt_ctrl_s=dt, warmup_s=warmup_s, dx_km=dx_km,
                          density=density, outflow=outflow)["return"]


def evaluate_rollouts(ensemble: DeepONetEnsemble, store_dir: str | Path, files: list[str], weights: RewardWeights,
                      warmup_s: float = 90.0, x_band_max_m: float = 1400.0, group_key: str = "controller_type") -> dict:
    store_dir = Path(store_dir)
    rows = []
    err_all, std_all = [], []
    per_k_err = np.zeros(ensemble.K); n_k = 0
    for fn in files:
        arrays, meta = load_rollout_npz(store_dir / fn)
        branch = ensemble.norm.branch_input(arrays["mainline_demand"], arrays["ramp_inflow_vph"])[None]
        rho_m, q_m = ensemble.predict_full(branch)           # (M, 1, Nx, K), (M, 1, K)
        rho_m, q_m = rho_m[:, 0], q_m[:, 0]
        rho_hat, q_hat = np.maximum(rho_m.mean(0), 0.0), np.maximum(q_m.mean(0), 0.0)
        rho_std, q_std = rho_m.std(0), q_m.std(0)
        true, q_true = arrays["density"].astype(np.float64), arrays["outflow_vph"].astype(np.float64)
        upstream = (ensemble.x_grid <= x_band_max_m)[:, None]
        free = true < FREE_MAX
        band = (true >= FREE_MAX) & (true <= JAM_MIN) & upstream
        jam = true > JAM_MIN
        dx_km = float(ensemble.x_grid[1] - ensemble.x_grid[0]) / 1000.0
        r_true = episode_return(arrays, true, q_true, weights, warmup_s, ensemble.dt, dx_km)
        r_pred = episode_return(arrays, rho_hat, q_hat, weights, warmup_s, ensemble.dt, dx_km)
        r_members = [episode_return(arrays, np.maximum(rho_m[m], 0), np.maximum(q_m[m], 0), weights, warmup_s, ensemble.dt, dx_km) for m in range(ensemble.M)]
        # the rollout's own threshold (scenario detectors.breakdown_density_veh_km, recorded in its metrics; M15)
        bd_thr = float(meta.get("metrics", {}).get("breakdown_density_veh_km", 60.0))
        bf_true = breakdown_flags(true, ensemble.dt, threshold=bd_thr); bf_pred = breakdown_flags(rho_hat, ensemble.dt, threshold=bd_thr)
        row = {
            "file": fn, "group": str(meta.get("controller", {}).get("type", "unknown")) if group_key == "controller_type" else str(meta.get(group_key, "")),
            "round": int(meta.get("round", 0)), "peak_total": float(meta.get("profile", {}).get("peak_total_vph", 0.0)),
            "rel_l2_density": _rel_l2(rho_hat, true), "rel_l2_free": _rel_l2(rho_hat, true, free),
            "rel_l2_band": _rel_l2(rho_hat, true, band), "rel_l2_jam": _rel_l2(rho_hat, true, jam),
            "rel_l2_flow": _rel_l2(q_hat, q_true), "outflow_sum_err": abs(q_hat.sum() - q_true.sum()) / max(q_true.sum(), 1e-6),
            "return_true": r_true, "return_pred": r_pred, "return_members": r_members,
            "return_err": abs(r_pred - r_true), "return_rel_err": abs(r_pred - r_true) / max(abs(r_true), 1e-6),
            "return_spread": float(np.std(r_members)),
            "breakdown_true": bf_true["breakdown"], "breakdown_pred": bf_pred["breakdown"],
            "onset_true_min": bf_true["breakdown_onset_min"], "onset_pred_min": bf_pred["breakdown_onset_min"],
            "onset_err_min": (bf_pred["breakdown_onset_min"] - bf_true["breakdown_onset_min"]) if (bf_true["breakdown"] and bf_pred["breakdown"]) else float("nan"),
            "mean_std_density": float(rho_std.mean()), "mean_abs_err_density": float(np.abs(rho_hat - true).mean()),
            "mean_std_flow": float(q_std.mean()), "mean_abs_err_flow": float(np.abs(q_hat - q_true).mean()),
            "frac_within_2std": float(np.mean(np.abs(rho_hat - true) <= 2.0 * rho_std + 1e-3)),
        }
        rows.append(row)
        err_all.append(np.abs(rho_hat - true).ravel()); std_all.append(rho_std.ravel())
        per_k_err += np.abs(rho_hat - true).mean(0); n_k += 1
    err = np.concatenate(err_all); std = np.concatenate(std_all)
    calib = calibration(err, std)
    summary = {"n": len(rows), "per_k_abs_err": (per_k_err / max(n_k, 1)).tolist(), "calibration": calib,
               "all": _aggregate(rows), "by_group": {}}
    for g in sorted({r["group"] for r in rows}):
        summary["by_group"][g] = _aggregate([r for r in rows if r["group"] == g])
    summary["gate"] = {
        "return_rel_err": summary["all"]["return_rel_err_mean"],
        "return_rel_err_ok": summary["all"]["return_rel_err_mean"] <= 0.10,
        "false_breakdown_rate": summary["all"]["false_breakdown_rate"],
        "false_breakdown_ok": summary["all"]["false_breakdown_rate"] <= 0.10,
        "calibration_slope": calib["slope"],
        "calibration_ok": 0.5 <= calib["slope"] <= 2.0,
    }
    summary["gate"]["passed"] = all(summary["gate"][k] for k in ("return_rel_err_ok", "false_breakdown_ok", "calibration_ok"))
    return {"rows": rows, "summary": summary}


def calibration(abs_err: np.ndarray, std: np.ndarray, n_bins: int = 10) -> dict:
    """Bin |error| by ensemble std; slope of |err| vs std through the origin,
    binned means, and the coverage of +-2 std."""
    std = np.asarray(std, dtype=np.float64); abs_err = np.asarray(abs_err, dtype=np.float64)
    keep = std > 1e-6
    if keep.sum() < 10:
        return {"slope": float("nan"), "bins": [], "coverage_2std": float("nan")}
    s, e = std[keep], abs_err[keep]
    slope = float(np.sum(s * e) / np.sum(s * s))
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        if m.any():
            bins.append({"std_mean": float(s[m].mean()), "abs_err_mean": float(e[m].mean()),
                         "abs_err_rms": float(np.sqrt(np.mean(e[m] ** 2))), "n": int(m.sum())})
    return {"slope": slope, "bins": bins, "coverage_2std": float(np.mean(abs_err <= 2.0 * std + 1e-3)),
            "corr": float(np.corrcoef(s, e)[0, 1]) if len(s) > 2 else float("nan")}


def _aggregate(rows: list[dict]) -> dict:
    def mean(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    n = len(rows)
    bt = np.array([r["breakdown_true"] for r in rows]); bp = np.array([r["breakdown_pred"] for r in rows])
    return {
        "n": n,
        "rel_l2_density": mean("rel_l2_density"), "rel_l2_free": mean("rel_l2_free"),
        "rel_l2_band": mean("rel_l2_band"), "rel_l2_jam": mean("rel_l2_jam"),
        "rel_l2_flow": mean("rel_l2_flow"), "outflow_sum_err": mean("outflow_sum_err"),
        "return_rel_err_mean": mean("return_rel_err"), "return_abs_err_mean": mean("return_err"),
        "return_rel_err_median": float(np.median([r["return_rel_err"] for r in rows])) if n else float("nan"),
        "return_spread_mean": mean("return_spread"),
        "breakdown_rate_true": float(bt.mean()) if n else float("nan"),
        "false_breakdown_rate": float(np.mean(bp[~bt])) if (~bt).any() else 0.0,
        "missed_breakdown_rate": float(np.mean(~bp[bt])) if bt.any() else 0.0,
        "onset_err_min_mean_abs": mean_abs_finite([r["onset_err_min"] for r in rows]),
        "mean_std_density": mean("mean_std_density"), "mean_abs_err_density": mean("mean_abs_err_density"),
        "frac_within_2std": mean("frac_within_2std"),
    }


def mean_abs_finite(values) -> float:
    v = np.array(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.abs(v).mean()) if len(v) else float("nan")


def write_report(result: dict, out_dir: str | Path, name: str = "surrogate_eval") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result["summary"], indent=1))
    with (out_dir / f"{name}_rows.jsonl").open("w") as f:
        for r in result["rows"]:
            f.write(json.dumps(r) + "\n")
    return out_dir / f"{name}.json"


def print_summary(summary: dict, title: str = "") -> None:
    print(f"\n== surrogate evaluation {title} (n={summary['n']}) ==")
    cols = ("n", "rel_l2_density", "rel_l2_free", "rel_l2_band", "rel_l2_jam", "rel_l2_flow", "return_rel_err_mean",
            "false_breakdown_rate", "missed_breakdown_rate", "onset_err_min_mean_abs", "frac_within_2std")
    print(f"  {'group':<16s}" + "".join(f"{c[:14]:>15s}" for c in cols))
    for g, a in [("ALL", summary["all"])] + list(summary["by_group"].items()):
        print(f"  {g:<16s}" + "".join(f"{a[c]:15.3f}" if isinstance(a[c], float) else f"{a[c]:15d}" for c in cols))
    gate = summary["gate"]
    print(f"  gate: return err {gate['return_rel_err']:.3f} (<=0.10 {gate['return_rel_err_ok']}), "
          f"false breakdown {gate['false_breakdown_rate']:.3f} (<=0.10 {gate['false_breakdown_ok']}), "
          f"calibration slope {gate['calibration_slope']:.2f} (in [0.5, 2] {gate['calibration_ok']}) -> "
          f"{'PASSED' if gate['passed'] else 'FAILED'}")
