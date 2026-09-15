"""
Closed-loop rollout driver (M8).

Rolls a controller through SumoEnv (or any env with the same step/info
contract) and returns the per-step arrays the plant-model dataset needs,
plus the episode metrics of draft_pipeline.md §10 (return, TTS, served
vehicles, final queue, breakdown / recovery flags). One function produces
the dataset labels, the aggregation rollouts and the evaluation numbers, so
the surrogate target and the policy observation are the same quantities by
construction.

npz contract (one file per rollout, draft "Rollout npz"):
    density          (N_x, K)  veh/km/lane, occupancy estimator
    speed, flow      (N_x, K)  diagnostics
    outflow_vph      (K,)      exact network-arrival count * 3600 / dt
    mainline_demand  (K,)      d_k
    ramp_arrival     (K,)      r_k
    ramp_inflow_vph  (K,)      q_r,k = confirmed ramp entries * 3600 / dt (branch channel 2)
    ramp_queue       (K,)      Q_k at interval end
    pending_mainline (K,)      diagnostic
    pending_ramp (K,)          released-but-not-inserted ramp vehicles (diagnostic; they stay in ramp_queue)
    action           (K,)      u_k applied (after the ramp storage cap, M14)
    action_requested (K,)      u_k asked for by the controller
    reward           (K,)      r_k as PPO sees it (warm-up masked)
    q_ref            (K,)      per-step outflow reference used by the reward
    x_grid, t_grid
    meta (json string): profile, controller spec, seed, speed_dev, metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BREAKDOWN_DENSITY = 60.0     # veh/km/lane, max over detectors
BREAKDOWN_MIN_STEPS = 10     # 5 min at 30 s steps


def breakdown_flags(density: np.ndarray, dt_ctrl_s: float = 30.0) -> dict:
    """Breakdown = max_i rho_{i,k} > 60 veh/km for >= 5 consecutive minutes.
    Recovered = a breakdown occurred and the last 5 minutes are jam-free."""
    rho_max = np.max(np.asarray(density, dtype=np.float64), axis=0)   # (K,)
    jam = rho_max > BREAKDOWN_DENSITY
    n_min = max(1, int(round(BREAKDOWN_MIN_STEPS * 30.0 / dt_ctrl_s)))
    onset = None
    run = 0
    for k, j in enumerate(jam):
        run = run + 1 if j else 0
        if run >= n_min and onset is None:
            onset = k - n_min + 1
    breakdown = onset is not None
    recovered = bool(breakdown and not jam[-n_min:].any())
    end = None
    if breakdown:
        after = np.where(~jam[onset:])[0]
        # first index after onset where a jam-free run of n_min starts
        k = onset
        while k < len(jam):
            if not jam[k:k + n_min].any() and (k + n_min) <= len(jam):
                end = k
                break
            k += 1
    return {
        "breakdown": bool(breakdown),
        "breakdown_onset_step": -1 if onset is None else int(onset),
        "breakdown_onset_min": -1.0 if onset is None else float(onset * dt_ctrl_s / 60.0),
        "breakdown_end_step": -1 if end is None else int(end),
        "recovered": recovered,
        "jam_steps": int(jam.sum()),
        "rho_max": float(rho_max.max()),
    }


def episode_metrics(arrays: dict, dt_ctrl_s: float, dx_km: float) -> dict:
    """Metrics of §10 from the per-step arrays."""
    density = np.asarray(arrays["density"], dtype=np.float64)
    K = density.shape[1]
    dt_h = dt_ctrl_s / 3600.0
    veh_on_road = density.sum(axis=0) * dx_km
    queue = np.asarray(arrays["ramp_queue"], dtype=np.float64)
    pending = np.asarray(arrays.get("pending_mainline", np.zeros(K)), dtype=np.float64)
    tts_veh_h = float(dt_h * np.sum(veh_on_road + queue + pending))
    served = float(np.sum(np.asarray(arrays["outflow_vph"], dtype=np.float64)) * dt_h)
    offered = float(np.sum(np.asarray(arrays["mainline_demand"]) + np.asarray(arrays["ramp_arrival"])) * dt_h)
    rewards = np.asarray(arrays["reward"], dtype=np.float64)
    out = {
        "return": float(rewards.sum()),
        "tts_veh_h": tts_veh_h,
        "served_veh": served,
        "offered_veh": offered,
        "final_queue": float(queue[-1]),
        "max_queue": float(queue.max()),
        "final_pending_mainline": float(pending[-1]),
        "pending_ramp_max": float(np.max(arrays.get("pending_ramp", np.zeros(K)))),     # peak blocked-at-the-meter count
        "pending_ramp_steps": int(np.sum(np.asarray(arrays.get("pending_ramp", np.zeros(K))) > 0.5)),
        "mean_action": float(np.mean(arrays["action"])),
        "queue_override_frac": float(np.mean(np.asarray(arrays["action"]) > np.asarray(arrays.get("action_requested", arrays["action"])) + 1e-6)),
        "outflow_penalty_sum": float(np.sum(arrays.get("outflow_penalty", 0.0))),
        "queue_penalty_sum": float(np.sum(arrays.get("queue_penalty", 0.0))),
        "std_penalty_sum": float(np.sum(arrays.get("std_penalty", 0.0))),
    }
    out.update(breakdown_flags(density, dt_ctrl_s))
    return out


def rollout_episode(env, controller, reset_options: dict | None = None, deterministic: bool = True) -> dict:
    """Roll `controller` through one episode of `env`.

    controller: callable. Called as controller(obs, info) if it accepts two
        arguments (behaviour controllers), else controller(obs). Optional
        .reset(env) / .reset() before the episode. Returns u in [0, 1] (array
        or float).
    Returns dict(arrays=..., metrics=..., info=<last info>, wall_s=...).
    """
    t0 = time.time()
    obs, info = env.reset(options=reset_options or {})
    _reset_controller(controller, env)
    K = env.T_ctrl
    N_x = env.N_x
    arr = {
        "density": np.zeros((N_x, K), np.float32),
        "speed": np.zeros((N_x, K), np.float32),
        "flow": np.zeros((N_x, K), np.float32),
        "outflow_vph": np.zeros(K, np.float32),
        "mainline_demand": np.zeros(K, np.float32),
        "ramp_arrival": np.zeros(K, np.float32),
        "ramp_inflow_vph": np.zeros(K, np.float32),
        "ramp_queue": np.zeros(K, np.float32),
        "pending_mainline": np.zeros(K, np.float32),
        "pending_ramp": np.zeros(K, np.float32),      # released but not yet inserted (ramp blocked by the jam, M14)
        "action": np.zeros(K, np.float32),            # applied metering rate (after the queue-cap override, M14)
        "action_requested": np.zeros(K, np.float32),  # what the controller asked for
        "reward": np.zeros(K, np.float32),
        "q_ref": np.zeros(K, np.float32),
        "outflow_penalty": np.zeros(K, np.float32),
        "queue_penalty": np.zeros(K, np.float32),
        "std_penalty": np.zeros(K, np.float32),
        "backlog_est": np.zeros(K, np.float32),
        "tts_step": np.zeros(K, np.float32),
        "obs": np.zeros((K, int(np.asarray(obs).shape[0])), np.float32),
    }
    extra = {}
    last_info = info
    for k in range(K):
        u = _call_controller(controller, obs, last_info)
        u = float(np.clip(np.asarray(u, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        arr["obs"][k] = obs
        obs, reward, terminated, truncated, last_info = env.step(np.array([u], dtype=np.float32))
        arr["density"][:, k] = last_info["density"]
        arr["speed"][:, k] = last_info.get("speed", np.zeros(N_x))
        arr["flow"][:, k] = last_info.get("flow", np.zeros(N_x))
        arr["outflow_vph"][k] = last_info["outflow_vph"]
        arr["mainline_demand"][k] = last_info.get("mainline_demand_vph", last_info.get("demand_vph", 0.0))
        arr["ramp_arrival"][k] = last_info.get("ramp_arrival_vph", last_info.get("ramp_demand_vph", 0.0))
        arr["ramp_inflow_vph"][k] = last_info.get("ramp_inflow_vph", 0.0)
        arr["ramp_queue"][k] = last_info.get("queue_after", last_info.get("analytical_queue", 0.0))
        arr["pending_mainline"][k] = last_info.get("pending_mainline", 0.0)
        arr["pending_ramp"][k] = last_info.get("pending_ramp", 0.0)
        arr["action"][k] = float(last_info.get("u", u))
        arr["action_requested"][k] = u
        arr["reward"][k] = reward
        arr["q_ref"][k] = last_info.get("q_ref", 0.0)
        arr["backlog_est"][k] = last_info.get("backlog_veh", 0.0)
        arr["tts_step"][k] = last_info.get("tts_step_veh_h", 0.0)
        masked = bool(last_info.get("reward_warmup_active", 0.0))
        for key in ("outflow_penalty", "queue_penalty", "std_penalty"):
            arr[key][k] = 0.0 if masked else float(last_info.get(key, 0.0))
        for key in ("ensemble_member", "ensemble_std_density", "ensemble_std_outflow"):
            if key in last_info:
                extra.setdefault(key, []).append(last_info[key])
        if terminated or truncated:
            break
    dx_km = float(getattr(env, "x_grid", np.arange(N_x) * 100.0)[1] - getattr(env, "x_grid", np.arange(N_x) * 100.0)[0]) / 1000.0 if N_x > 1 else 0.1
    metrics = episode_metrics(arr, float(env.dt_ctrl), dx_km)
    metrics["wall_s"] = float(time.time() - t0)
    metrics["teleports"] = int(last_info.get("teleports", 0))
    metrics["discarded_mainline"] = int(last_info.get("discarded_mainline", 0))
    for key, vals in extra.items():
        metrics[key + "_mean"] = float(np.mean(vals))
    return {"arrays": arr, "metrics": metrics, "info": last_info, "wall_s": metrics["wall_s"]}


def rescore_return(arrays: dict, weights, dt_ctrl_s: float = 30.0, warmup_s: float = 90.0, dx_km: float = 0.1,
                   density: np.ndarray | None = None, outflow: np.ndarray | None = None) -> dict:
    """Episode return of stored (or predicted) fields under any reward weights.

    Recomputes every per-step term from density (Nx, K), outflow (K), ramp
    queue (K), demand (K), ramp arrivals (K) and the stored q_ref (K), with
    the conservation backlog and the warm-up mask, exactly as the envs do.
    """
    from rl.reward import backlog_estimate, reward_terms

    dens = np.asarray(arrays["density"] if density is None else density, dtype=np.float64)
    out = np.asarray(arrays["outflow_vph"] if outflow is None else outflow, dtype=np.float64)
    queue = np.asarray(arrays["ramp_queue"], dtype=np.float64)
    d = np.asarray(arrays["mainline_demand"], dtype=np.float64); r = np.asarray(arrays["ramp_arrival"], dtype=np.float64)
    q_ref = arrays.get("q_ref")
    K = dens.shape[1]
    dt_h = dt_ctrl_s / 3600.0
    warm_steps = int(round(warmup_s / dt_ctrl_s))
    cum_off = cum_srv = 0.0
    total = 0.0; parts = {"outflow_penalty": 0.0, "queue_penalty": 0.0, "std_penalty": 0.0, "tts_penalty": 0.0, "terminal_penalty": 0.0}
    for k in range(K):
        cum_off += (d[k] + r[k]) * dt_h; cum_srv += max(out[k], 0.0) * dt_h
        on_road = float(np.sum(np.maximum(dens[:, k], 0.0))) * dx_km
        bl = backlog_estimate(cum_off, cum_srv, on_road, queue[k])
        t = reward_terms(np.maximum(dens[:, k], 0.0), float(max(queue[k], 0.0)), float(max(out[k], 0.0)), weights,
                         q_ref=None if q_ref is None else float(q_ref[k]), terminal=(k == K - 1), backlog_veh=bl, dt_ctrl_s=dt_ctrl_s)
        if k >= warm_steps:
            total += t["reward"]
            for key in parts:
                parts[key] += t[key]
    return {"return": float(total), **{k: float(v) for k, v in parts.items()}}


def _reset_controller(controller, env) -> None:
    reset = getattr(controller, "reset", None)
    if reset is None:
        return
    try:
        reset(env)
    except TypeError:
        reset()


def _call_controller(controller, obs, info):
    try:
        return controller(obs, info)
    except TypeError:
        return controller(obs)


def save_rollout_npz(path: str | Path, result: dict, meta: dict, env=None) -> Path:
    """Write the npz contract; `meta` is stored as a JSON string under 'meta'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = result["arrays"]
    payload = {k: v for k, v in arr.items() if k != "obs"}
    meta = dict(meta)
    meta["metrics"] = result["metrics"]
    if env is not None:
        payload["x_grid"] = np.asarray(env.x_grid, dtype=np.float32)
        payload["t_grid"] = np.asarray(env.t_grid, dtype=np.float32)
        meta.setdefault("dt_ctrl_s", float(env.dt_ctrl))
        meta.setdefault("highway_length_m", float(env.sumo_config["network"]["highway_length_m"]))
    payload["meta"] = np.array(json.dumps(_jsonable(meta)))
    np.savez_compressed(str(path), **payload)
    return path


def load_rollout_npz(path: str | Path) -> tuple[dict, dict]:
    """Return (arrays, meta) from a rollout file written by save_rollout_npz."""
    with np.load(str(path)) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return arrays, meta


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj
