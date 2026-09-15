"""
Evaluate policies on frozen profile sets in SUMO (M11/M12, draft §10).

`evaluate_on_profiles` rolls every (policy, profile, seed) episode through
SumoEnv with the parallel runner, writes one JSON line per episode, appends
ledger lines (purpose eval_val / eval_test / eval_ood / aggregation /
tuning) and returns a per-policy summary with the §10 metrics:

    mean return, 10th percentile, worst episode, breakdown rate, recovery
    rate, TTS (veh h), served vehicles, final queue,

plus paired bootstrap 95 % intervals of the mean-return difference against a
reference policy over the same (profile, seed) episodes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rl.policy_specs import policy_label
from sumo_env.demand_profiles import DemandProfile, load_profile_set
from sumo_env.parallel_rollouts import run_jobs
from utils.config import load_config, merge_configs
from utils.ledger import Ledger


def sumo_env_config(project_root: Path, config: str = "configs/rl/ppo_common.yaml", overlay: str = "configs/rl/env_sumo.yaml",
                    network_dir: str | None = None, extra: dict | None = None) -> dict:
    """The env block of ppo_common + env_sumo (density stats resolved)."""
    cfg = load_config(str(project_root / config))
    if overlay:
        cfg = merge_configs(cfg, load_config(str(project_root / overlay)))
    env_cfg = dict(cfg["env"])
    env_cfg["project_root"] = str(project_root)
    if extra:
        env_cfg = merge_configs(env_cfg, extra)
    if network_dir:
        env_cfg["network_dir"] = network_dir
    if env_cfg.get("density_mean") in ("auto", None) or env_cfg.get("density_std") in ("auto", None):
        src = env_cfg.get("density_stats_from")
        if env_cfg.get("ensemble_dir") and (project_root / env_cfg["ensemble_dir"] / "manifest.json").exists():
            stats = json.loads((project_root / env_cfg["ensemble_dir"] / "manifest.json").read_text())["normalization"]
        elif src and (project_root / src).exists():
            data = json.loads((project_root / src).read_text())
            stats = data.get("metadata", data)
        else:
            raise ValueError("density_mean 'auto' needs env.density_stats_from or env.ensemble_dir")
        env_cfg["density_mean"] = float(stats["mean_density"]); env_cfg["density_std"] = float(stats["std_density"])
    return env_cfg


def evaluate_on_profiles(policies: list[str], profiles: list[DemandProfile], env_cfg: dict, out_jsonl: Path,
                         seeds: list[int] | None = None, workers: int = 8, purpose: str = "eval_val", study: str = "eval",
                         round_index: int = 0, save_rollouts_dir: Path | None = None, project_root: Path | None = None,
                         network_root: str | None = None, labels: dict | None = None, quiet: bool = False) -> dict:
    root = Path(project_root or Path.cwd())
    out_jsonl = Path(out_jsonl); out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    labels = labels or {}
    jobs = []
    for pi, policy in enumerate(policies):
        for p in profiles:
            ep_seeds = seeds if seeds is not None else (p.sumo_seeds or [10000 + p.index])
            for s in ep_seeds:
                jobs.append({"name": f"{purpose}_p{pi:02d}_{p.set_name}{p.index:03d}_s{s}", "profile": p.to_dict(),
                             "sumo_seed": int(s), "policy": str(policy), "policy_index": pi, "round": round_index,
                             "purpose": purpose, "study": study,
                             "out_dir": str(save_rollouts_dir) if save_rollouts_dir else None})
    t0 = time.time()
    results = run_jobs(jobs, env_cfg, workers=workers, network_root=network_root, progress=not quiet)
    ledger = Ledger(study, root)
    rows = []
    with out_jsonl.open("a", encoding="utf-8") as f:
        for job, res in zip(jobs, results):
            m = res["metrics"]
            row = {"policy": job["policy"], "label": labels.get(job["policy"], policy_label(job["policy"])),
                   "profile_set": job["profile"]["set"], "profile_index": job["profile"]["index"],
                   "peak_total_vph": job["profile"].get("peak_total_vph"), "sumo_seed": job["sumo_seed"],
                   "purpose": purpose, "round": round_index, "study": study, "file": res.get("path"), **m}
            f.write(json.dumps(row) + "\n"); rows.append(row)
            ledger.log(round_index, purpose, job["profile"]["set"], job["profile"]["index"], job["sumo_seed"],
                       job["policy"], m["return"], m["breakdown"], m["wall_s"])
    summary = summarise_rows(rows, reference=policies[0] if policies else None)
    summary["wall_s"] = time.time() - t0
    summary["n_episodes"] = len(rows)
    out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return {"rows": rows, "summary": summary, "results": results}


def summarise_rows(rows: list[dict], reference: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    summary: dict = {"policies": {}, "reference": reference}
    for policy, eps in by_policy.items():
        ret = np.array([e["return"] for e in eps], dtype=np.float64)
        summary["policies"][policy] = {
            "label": eps[0]["label"], "n": len(eps),
            "mean": float(ret.mean()), "std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
            "p10": float(np.percentile(ret, 10)), "worst": float(ret.min()), "best": float(ret.max()),
            "breakdown_rate": float(np.mean([e["breakdown"] for e in eps])),
            "recovery_rate": float(np.mean([e["recovered"] for e in eps if e["breakdown"]])) if any(e["breakdown"] for e in eps) else float("nan"),
            "tts_veh_h": float(np.mean([e["tts_veh_h"] for e in eps])),
            "served_veh": float(np.mean([e["served_veh"] for e in eps])),
            "final_queue": float(np.mean([e["final_queue"] for e in eps])),
            "mean_action": float(np.mean([e["mean_action"] for e in eps])),
            "n_catastrophic": int(np.sum(ret < -150.0)),
        }
    if reference in by_policy:
        rng = np.random.default_rng(seed)
        ref = {(e["profile_set"], e["profile_index"], e["sumo_seed"]): e["return"] for e in by_policy[reference]}
        for policy, eps in by_policy.items():
            if policy == reference:
                continue
            pairs = [(e["return"], ref[(e["profile_set"], e["profile_index"], e["sumo_seed"])]) for e in eps
                     if (e["profile_set"], e["profile_index"], e["sumo_seed"]) in ref]
            if not pairs:
                continue
            diff = np.array([a - b for a, b in pairs])
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
            summary["policies"][policy]["paired_diff_vs_reference"] = {
                "mean": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "n_pairs": len(pairs), "win_rate": float(np.mean(diff > 0))}
    return summary


def print_summary(summary: dict, title: str = "") -> None:
    print(f"\n== {title} ==")
    print(f"  {'policy':<44s} {'n':>3} {'mean':>8} {'p10':>8} {'worst':>8} {'bd':>5} {'rec':>5} {'TTS':>7} {'served':>7} {'Qend':>6} {'u':>5}  diff vs ref [95% CI]")
    for policy, s in summary["policies"].items():
        d = s.get("paired_diff_vs_reference")
        dtxt = f"{d['mean']:+7.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d else "(reference)"
        print(f"  {s['label'][:44]:<44s} {s['n']:3d} {s['mean']:8.1f} {s['p10']:8.1f} {s['worst']:8.1f} {s['breakdown_rate']:5.2f} "
              f"{s['recovery_rate']:5.2f} {s['tts_veh_h']:7.1f} {s['served_veh']:7.0f} {s['final_queue']:6.0f} {s['mean_action']:5.2f}  {dtxt}")


def load_set(name_or_path: str, project_root: Path) -> list[DemandProfile]:
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = project_root / "configs" / "profiles" / f"{name_or_path}.json"
    elif not p.is_absolute():
        p = project_root / p
    return load_profile_set(p)
