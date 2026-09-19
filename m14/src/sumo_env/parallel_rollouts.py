"""
Parallel SUMO rollout runner (M8).

`run_jobs` executes a list of rollout jobs across worker processes, each
owning one SumoEnv (private network directory, private TraCI label). Jobs are
dicts:

    {"name": "r0_constant_00012",          # npz stem
     "profile": {"set": "train", "draw": 17}   # sample from the env's family, or a DemandProfile dict
     "sumo_seed": 50012,
     "controller": {...},                  # rl.behaviour_controllers spec, or
     "policy": "runs/.../ckpt.zip" | "u=0.3" | "alinea:..." | "mpc:...",   # eval-style policy spec
     "round": 0, "purpose": "dataset", ...any extra meta}

Results are saved with sumo_env.rollout.save_rollout_npz and the metadata
(profile, controller, seed, metrics) is returned to the caller for the store
and the ledger. Workers are spawned (not forked) so TraCI state is never
shared.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

_ENV = None
_ENV_CFG = None
_POLICY_CACHE: dict = {}


def _worker_init(env_cfg: dict, network_root: str) -> None:
    global _ENV, _ENV_CFG
    import sys

    root = Path(env_cfg.get("project_root", Path(__file__).resolve().parents[2]))
    for sub in ("src", "scripts"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch

    torch.set_num_threads(1)
    from rl.sumo_env_wrapper import SumoEnv

    ident = mp.current_process()._identity
    wid = ident[0] if ident else 0
    cfg = dict(env_cfg)
    cfg["network_dir"] = f"{network_root}_w{wid}"
    _ENV_CFG = cfg
    _ENV = SumoEnv(cfg)
    import atexit

    atexit.register(lambda: _ENV.close() if _ENV is not None else None)


def _resolve_profile(job: dict):
    from sumo_env.demand_profiles import DemandProfile

    spec = job.get("profile")
    if spec is None:
        return None
    if isinstance(spec, DemandProfile):
        return spec
    if "mainline_vph_blocks" in spec:
        return DemandProfile.from_dict(spec)
    if _ENV.profile_family is None:
        raise ValueError("job asks for a family draw but the env has no profile family")
    return _ENV.profile_family.sample_by_key(str(spec.get("set", "train")), int(spec["draw"]))


def _build_controller(job: dict, profile):
    from rl.behaviour_controllers import make_controller_from_spec

    if "controller" in job and job["controller"] is not None:
        return make_controller_from_spec(job["controller"], _ENV, profile)
    policy = job.get("policy")
    if policy is None:
        raise ValueError("job needs 'controller' or 'policy'")
    from rl.policy_specs import make_policy_callable

    key = str(policy)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = make_policy_callable(key, _ENV, _ENV_CFG)
    return _POLICY_CACHE[key]


def _run_job(job: dict) -> dict:
    from sumo_env.rollout import rollout_episode, save_rollout_npz

    t0 = time.time()
    try:
        profile = _resolve_profile(job)
        controller = _build_controller(job, profile)
        options = {"sumo_seed": int(job.get("sumo_seed", 0))}
        if profile is not None:
            options["profile"] = profile
        for key in ("demand_vph", "ramp_demand_vph"):
            if key in job:
                options[key] = float(job[key])
        result = rollout_episode(_ENV, controller, reset_options=options)
        meta = {k: v for k, v in job.items() if k not in ("profile", "out_dir")}
        meta["profile"] = profile.to_dict() if profile is not None else {"set": "grid", "index": -1}
        if profile is not None:
            meta["profile"]["peak_total_vph"] = profile.peak_total_vph
        if "controller" not in meta or meta["controller"] is None:
            meta["controller"] = getattr(controller, "spec", {"type": "policy", "policy": str(job.get("policy"))})
        meta["speed_dev"] = float(_ENV.sumo_config.get("vehicle", {}).get("speed_dev", 0.0))
        meta["density_method"] = _ENV.density_method
        meta["merge_station_lanes"] = getattr(_ENV, "merge_station_lanes", "mean")
        meta["wall_s"] = float(time.time() - t0)
        out_dir = job.get("out_dir")
        path = None
        if out_dir:
            path = save_rollout_npz(Path(out_dir) / f"{job['name']}.npz", result, meta, env=_ENV)
        meta["metrics"] = result["metrics"]
        return {"ok": True, "name": job["name"], "path": str(path) if path else None, "meta": meta,
                "metrics": result["metrics"],
                "arrays": result["arrays"] if job.get("return_arrays") else None}
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        return {"ok": False, "name": job.get("name"), "error": f"{exc}\n{traceback.format_exc()}"}


def run_jobs(jobs: list[dict], env_cfg: dict, workers: int = 8, network_root: str | None = None,
             progress: bool = True, chunksize: int = 1) -> list[dict]:
    """Run all jobs; returns results in job order. Raises on the first failure."""
    if not jobs:
        return []
    network_root = network_root or str(Path(env_cfg.get("network_dir", "data/networks/workers")))
    workers = max(1, min(int(workers), len(jobs)))
    t0 = time.time()
    results: list[dict] = [None] * len(jobs)  # type: ignore
    if workers == 1:
        _worker_init(env_cfg, network_root)
        for i, job in enumerate(jobs):
            results[i] = _run_job(job)
            if progress:
                _print_progress(i + 1, len(jobs), results[i], t0)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_worker_init, initargs=(env_cfg, network_root))
        try:
            for i, res in enumerate(pool.imap(_run_job, jobs, chunksize=chunksize)):
                results[i] = res
                if progress:
                    _print_progress(i + 1, len(jobs), res, t0)
            pool.close()     # let the workers exit normally (atexit closes their SUMO)
            pool.join()
        finally:
            pool.terminate()
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise RuntimeError(f"{len(failed)} rollout job(s) failed; first: {failed[0]['name']}\n{failed[0]['error']}")
    return results


def _print_progress(i: int, n: int, res: dict, t0: float) -> None:
    if not res.get("ok"):
        print(f"  [{i:>4}/{n}] {res.get('name')} FAILED", flush=True)
        return
    m = res["metrics"]
    el = time.time() - t0
    print(f"  [{i:>4}/{n}] {res['name']:<32s} return={m['return']:8.1f} tts={m['tts_veh_h']:6.1f} "
          f"bd={int(m['breakdown'])} rec={int(m['recovered'])} Qend={m['final_queue']:5.0f} "
          f"({el:5.0f}s, {el / i:4.1f} s/ep)", flush=True)
