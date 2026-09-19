"""
Evaluate policies on frozen M14 profile sets in SUMO.

`evaluate_on_profiles` rolls every (policy, profile, seed) episode through
SumoEnv with the parallel runner, atomically writes one JSON line per episode, appends
ledger lines (purpose eval_val / eval_test / eval_ood / aggregation /
tuning) and returns a per-policy summary with the following metrics:

    mean return, 10th percentile, worst episode, breakdown rate, recovery
    rate, TTS (veh h), served vehicles, final queue,

plus paired bootstrap 95 % intervals of the mean-return difference against a
reference policy over the same (profile, seed) episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tempfile
from pathlib import Path

import numpy as np

from rl.policy_specs import policy_label
from sumo_env.demand_profiles import DemandProfile, load_profile_set
from sumo_env.parallel_rollouts import run_jobs
from utils.config import load_config, merge_configs
from utils.ledger import Ledger


def scenario_overlays() -> list[str]:
    """Optional environment overlays, separated by ':' or whitespace.

    Empty/unset selects no overlays: configs/ppo.yaml already contains M14 v3b.
    """
    raw = os.environ.get("SCENARIO_OVERLAY", "").replace(":", " ").split()
    return [r for r in raw if r]


def sumo_env_config(project_root: Path, config: str = "configs/ppo.yaml", overlay: str = "configs/env_sumo.yaml",
                    network_dir: str | None = None, extra: dict | None = None, scenario_overlay: list[str] | None = None) -> dict:
    """The env block of ppo + env_sumo (+ the scenario overlays; density stats resolved)."""
    cfg = load_config(str(project_root / config))
    if overlay:
        cfg = merge_configs(cfg, load_config(str(project_root / overlay)))
    for ov in (scenario_overlays() if scenario_overlay is None else scenario_overlay):
        cfg = merge_configs(cfg, load_config(str(project_root / ov)))
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


def _canonical_json(value) -> str:
    """Stable JSON for request identities, including NumPy config values."""
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported evaluation request value: {type(obj).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=convert)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    """Publish a complete file, never a truncated JSONL or sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _request_dependencies(policies: list[str], env_cfg: dict, root: Path) -> dict:
    """Hash policy weights and referenced configuration, not just their paths."""
    files = {}

    def resolve(value) -> Path:
        p = Path(value)
        return (p if p.is_absolute() else root / p).resolve()

    def add_file(value, required=True):
        p = resolve(value)
        if not p.is_file():
            if required:
                raise FileNotFoundError(f"Evaluation dependency is missing: {p}")
            files[str(p)] = None
            return
        files[str(p)] = _file_sha256(p)

    def add_ensemble(value):
        directory = resolve(value)
        manifest = directory / "manifest.json"
        if manifest.is_file():
            add_file(manifest)
            members = [directory / name for name in json.loads(manifest.read_text())["members"]]
        else:
            members = sorted(directory.glob("member_*/best.pt"))
        if not members:
            raise FileNotFoundError(f"Evaluation ensemble has no member checkpoints: {directory}")
        for member in members:
            add_file(member)

    for policy in policies:
        spec = str(policy)
        head, _, tail = spec.partition(":")
        if spec.startswith("u=") or head.lower() in ("alinea", "pialinea"):
            continue
        if head.lower() == "mpc":
            add_ensemble(tail.split(",", 1)[0])
        else:
            add_file(spec)
    for key in ("sumo_config", "density_stats_from"):
        if env_cfg.get(key):
            # Explicit numeric statistics can make a stats source optional.
            add_file(env_cfg[key], required=(key == "sumo_config"))
    profile_source = env_cfg.get("profiles")
    if isinstance(profile_source, dict):
        for key in ("family", "set"):
            if profile_source.get(key):
                add_file(profile_source[key])
    elif isinstance(profile_source, (str, Path)):
        add_file(profile_source)
    if env_cfg.get("ensemble_dir"):
        add_ensemble(env_cfg["ensemble_dir"])
    return files


def _episode_key(row: dict) -> tuple:
    return (row["policy"], row["profile_set"], int(row["profile_index"]), int(row["sumo_seed"]))


def _job_key(job: dict) -> tuple:
    return (job["policy"], job["profile"]["set"], int(job["profile"]["index"]), int(job["sumo_seed"]))


def _restore_results(rows: list[dict], jobs: list[dict]) -> list[dict]:
    """Rebuild worker-shaped results, including saved metadata for aggregation."""
    from sumo_env.rollout import load_rollout_npz

    results = []
    row_metadata = {"policy", "label", "profile_set", "profile_index", "peak_total_vph",
                    "sumo_seed", "purpose", "round", "study", "file"}
    for row, job in zip(rows, jobs):
        if row.get("file"):
            _, meta = load_rollout_npz(row["file"])
            metrics = meta["metrics"]
            profile = meta["profile"]
            if (str(meta.get("policy")), profile["set"], int(profile["index"]), int(meta["sumo_seed"])) != _job_key(job):
                raise ValueError(f"Saved rollout metadata does not match requested episode: {row['file']}")
            if float(metrics["return"]) != float(row["return"]):
                raise ValueError(f"Saved rollout return disagrees with report: {row['file']}")
        else:
            metrics = {k: v for k, v in row.items() if k not in row_metadata}
            meta = {k: v for k, v in job.items() if k != "out_dir"}
            meta["metrics"] = metrics
        results.append({"ok": True, "name": job["name"], "path": row.get("file"),
                        "meta": meta, "metrics": metrics, "arrays": None})
    return results


def evaluate_on_profiles(policies: list[str], profiles: list[DemandProfile], env_cfg: dict, out_jsonl: Path,
                         seeds: list[int] | None = None, workers: int = 8, purpose: str = "eval_val", study: str = "eval",
                         round_index: int = 0, save_rollouts_dir: Path | None = None, project_root: Path | None = None,
                         network_root: str | None = None, labels: dict | None = None, quiet: bool = False) -> dict:
    """Evaluate once, or reuse an intact result for the exact same request.

    A sidecar fingerprints policies, profiles, configuration and referenced files.
    Existing incomplete, malformed or different requests are refused; use a new
    output path after inspecting those artifacts. Reuse never adds ledger rows.
    """
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    out_jsonl = Path(out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = root / out_jsonl
    out_jsonl = out_jsonl.resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if save_rollouts_dir is not None:
        save_rollouts_dir = Path(save_rollouts_dir)
        if not save_rollouts_dir.is_absolute():
            save_rollouts_dir = root / save_rollouts_dir
        save_rollouts_dir = save_rollouts_dir.resolve()
    labels = labels or {}
    jobs = []
    for pi, policy in enumerate(policies):
        for p in profiles:
            ep_seeds = seeds if seeds is not None else (p.sumo_seeds or [10000 + p.index])
            for seed in ep_seeds:
                jobs.append({"name": f"{purpose}_p{pi:02d}_{p.set_name}{p.index:03d}_s{seed}", "profile": p.to_dict(),
                             "sumo_seed": int(seed), "policy": str(policy), "policy_index": pi, "round": round_index,
                             "purpose": purpose, "study": study,
                             "out_dir": str(save_rollouts_dir) if save_rollouts_dir else None})
    keys = [_job_key(job) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Evaluation request contains duplicate (policy, profile, seed) episodes")
    request = {"format_version": 1, "jobs": jobs, "environment": env_cfg,
               "network_root": network_root, "labels": labels,
               "dependencies": _request_dependencies(policies, env_cfg, root)}
    fingerprint = hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()
    sidecar = out_jsonl.with_suffix(".request.json")
    summary_path = out_jsonl.with_suffix(".summary.json")
    existing = any(p.exists() for p in (out_jsonl, sidecar, summary_path))
    if existing:
        try:
            marker = json.loads(sidecar.read_text())
            if not isinstance(marker, dict):
                raise ValueError("request sidecar is not a JSON object")
            if marker.get("format_version") != 1 or marker.get("fingerprint") != fingerprint:
                raise ValueError("request fingerprint differs (policies, profiles, configuration or file contents changed)")
            if marker.get("status") != "complete":
                raise ValueError("prior evaluation is incomplete")
            if _file_sha256(out_jsonl) != marker["output_sha256"]:
                raise ValueError("JSONL is incomplete or its contents changed")
            if _file_sha256(summary_path) != marker["summary_sha256"]:
                raise ValueError("summary is incomplete or its contents changed")
            rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
            if [_episode_key(row) for row in rows] != keys:
                raise ValueError("JSONL episode keys are missing, duplicated or unexpected")
            for row in rows:
                if row["study"] != study or row["purpose"] != purpose or int(row["round"]) != round_index:
                    raise ValueError("JSONL episode context differs from the request")
            expected_files = {str(row["file"]) for row in rows if row.get("file")}
            if not isinstance(marker["rollout_sha256"], dict):
                raise ValueError("saved rollout inventory is malformed")
            if set(marker["rollout_sha256"]) != expected_files:
                raise ValueError("saved rollout inventory differs from the JSONL")
            for name, digest in marker["rollout_sha256"].items():
                if _file_sha256(Path(name)) != digest:
                    raise ValueError(f"saved rollout contents changed: {name}")
            summary = json.loads(summary_path.read_text())
            if summary["n_episodes"] != len(jobs):
                raise ValueError("summary episode count differs from the request")
            results = _restore_results(rows, jobs)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(f"Cannot reuse evaluation {out_jsonl}: {exc}. "
                               "Existing artifacts were preserved; inspect them and use a new output path.") from exc
        if not quiet:
            print(f"Reusing verified evaluation: {out_jsonl} ({len(rows)} episodes)")
        return {"rows": rows, "summary": summary, "results": results}

    # A durable running marker prevents an interrupted request being mistaken
    # for an empty destination and replaying already accounted SUMO episodes.
    marker = {"format_version": 1, "fingerprint": fingerprint, "status": "running", "request": request}
    try:
        # Exclusive creation also prevents two simultaneous invocations from
        # both launching the same episodes before either publishes results.
        with sidecar.open("x", encoding="utf-8") as f:
            f.write(_canonical_json(marker) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"Evaluation request already exists: {sidecar}; another process may be running it") from exc
    t0 = time.time()
    results = run_jobs(jobs, env_cfg, workers=workers, network_root=network_root, progress=not quiet)
    if len(results) != len(jobs) or any(not result.get("ok", True) for result in results):
        raise RuntimeError(f"Incomplete evaluation worker results for {out_jsonl}; artifacts were preserved")
    rows = []
    for job, result in zip(jobs, results):
        metrics = result["metrics"]
        if save_rollouts_dir is not None and not result.get("path"):
            raise RuntimeError(f"Worker omitted a requested saved rollout: {job['name']}")
        rows.append({"policy": job["policy"], "label": labels.get(job["policy"], policy_label(job["policy"])),
                     "profile_set": job["profile"]["set"], "profile_index": job["profile"]["index"],
                     "peak_total_vph": job["profile"].get("peak_total_vph"), "sumo_seed": job["sumo_seed"],
                     "purpose": purpose, "round": round_index, "study": study, "file": result.get("path"), **metrics})
    summary = summarise_rows(rows, reference=policies[0] if policies else None)
    summary["wall_s"] = time.time() - t0
    summary["n_episodes"] = len(rows)
    rollout_hashes = {str(row["file"]): _file_sha256(Path(row["file"])) for row in rows if row.get("file")}
    if save_rollouts_dir is not None:
        _restore_results(rows, jobs)  # validate saved metadata before publication
    _atomic_text(out_jsonl, "".join(json.dumps(row) + "\n" for row in rows))
    _atomic_text(summary_path, json.dumps(summary, indent=1) + "\n")
    ledger = Ledger(study, root)
    for row in rows:
        ledger.log(round_index, purpose, row["profile_set"], row["profile_index"], row["sumo_seed"],
                   row["policy"], row["return"], row["breakdown"], row["wall_s"])
    marker.update(status="complete", output_sha256=_file_sha256(out_jsonl),
                  summary_sha256=_file_sha256(summary_path), rollout_sha256=rollout_hashes)
    _atomic_text(sidecar, _canonical_json(marker) + "\n")
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
    """A frozen profile set by name ('val', 'test', 'ood' → $PROFILE_SETS_DIR/<name>.json,
    default configs/profiles contains the frozen M14 demand sets) or by path."""
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = project_root / os.environ.get("PROFILE_SETS_DIR", "configs/profiles") / f"{name_or_path}.json"
    elif not p.is_absolute():
        p = project_root / p
    return load_profile_set(p)
