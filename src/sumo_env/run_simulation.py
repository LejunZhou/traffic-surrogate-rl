"""
Launch a single SUMO simulation via TraCI and collect trajectory data.

Each call to run_simulation:
1. Starts SUMO (headless) with the given network / route / detector files.
2. Steps through the simulation in 1 s increments.
3. Every dt_ctrl_s seconds (one control step), applies the ramp metering
   signal and reads the configured induction-loop detectors.
4. Returns structured numpy arrays matching the Phase 1 dataset schema.

Ramp metering control mechanism:
  Metered queue: arrivals wait upstream of SUMO; u * ramp_discharge_vph
  determines insertion request capacity. Pending requests remain in the queue
  until getDepartedIDList confirms entry onto the road, and are not resubmitted.
  Open loop: request vehicles at u * ramp_demand_vph without an upstream queue.
  Both models measure ramp inflow from confirmed entries in each control interval.

Density derivation from E1 induction-loop readings:
  Primary:  ρ [veh/km] = flow [veh/hr] / speed [km/hr]   (fundamental relation)
  Fallback: ρ [veh/km] = occupancy_fraction × 1000 / vehicle_length_m
            (used when mean speed < 5 km/h, i.e., stopped/very slow traffic)

Requires SUMO ≥ 1.18 with TraCI Python bindings on PYTHONPATH.
"""

from __future__ import annotations

import numpy as np

try:
    import traci
except ImportError as exc:
    raise ImportError(
        "TraCI not found. Install SUMO ≥ 1.18 and ensure the SUMO Python "
        "bindings are on PYTHONPATH (e.g. export PYTHONPATH=$SUMO_HOME/tools)."
    ) from exc

from sumo_env.detectors import get_detector_ids, get_detector_ids_per_lane, get_x_grid
from sumo_env.ramp_queue import MeteredRampQueue


def run_simulation(
    net_file: str,
    route_file: str,
    detector_file: str,
    ramp_control: np.ndarray,
    config: dict,
) -> dict:
    """Run one SUMO simulation episode and return trajectory data.

    Args:
        net_file: Absolute path to the compiled .net.xml file.
        route_file: Absolute path to the .rou.xml file.
        detector_file: Absolute path to the .add.xml detector file.
        ramp_control: shape (T_ctrl,), metering rate at each control step ∈ [0, 1].
        config: Full experiment config dict.

    Returns:
        Dict with keys:
            "density":         np.float32, shape (N_x, T_ctrl), veh/km
            "speed":           np.float32, shape (N_x, T_ctrl), km/h
            "flow":            np.float32, shape (N_x, T_ctrl), veh/hr
            "exit_boundary_flow_vph": np.float32, shape (T_ctrl,), mean flow of the last 3 detectors
            "x_grid":          np.float32, shape (N_x,),        metres from upstream
            "t_grid":          np.float32, shape (T_ctrl,),     seconds (start of each interval)
            "mainline_demand": np.float32, shape (T_ctrl,),     veh/hr (constant for this profile)
            "ramp_control":    np.float32, shape (T_ctrl,), confirmed inflow / reference (command in open_loop)
            "ramp_departed_count": np.int32, shape (T_ctrl,), confirmed ramp entries per interval
            "ramp_pending_count": np.int32, shape (T_ctrl,), pending ramp requests at interval end
            "ramp_inflow_vph": np.float32, shape (T_ctrl,), confirmed ramp entries / interval hours
            "ramp_queue":      np.float32, shape (T_ctrl,), arrivals minus confirmed entries at interval end (0 in open_loop)
            "metadata":        dict        (seed, demand_profile, config snapshot)
    """
    sim_cfg = config["simulation"]
    det_cfg = config["detectors"]
    demand_cfg = config["demand"]

    step_len: float = sim_cfg["step_length_s"]        # 1.0 s
    dt_ctrl: int = sim_cfg["dt_ctrl_s"]               # 30 s
    dt_ctrl_steps: int = int(dt_ctrl / step_len)      # 30 sub-steps per control step
    T_ctrl: int = int(sim_cfg["duration_s"] / dt_ctrl)  # 120
    warmup_s: float = float(sim_cfg.get("ramp_warmup_s", 0.0))
    seed: int = sim_cfg["seed"]
    sumo_binary: str = sim_cfg["sumo_binary"]

    ramp_demand_vph: float = demand_cfg["ramp_demand_vph"]
    # Ramp model (M7 §7.13): "open_loop" inserts ramp_control * ramp_demand_vph
    # directly (M2 behaviour, inflow <= ramp_demand_vph); "metered_queue" mirrors
    # SumoEnv — arrivals at ramp_demand_vph wait in a virtual queue and the meter
    # releases min(u * ramp_discharge_vph, queue), so inflow can exceed the
    # arrival rate (up to ramp_discharge_vph) while a queue exists.
    ramp_model: str = str(demand_cfg.get("ramp_model", "open_loop"))
    if ramp_model not in ("open_loop", "metered_queue"):
        raise ValueError(f"demand.ramp_model must be 'open_loop' or 'metered_queue', got {ramp_model!r}")
    ramp_discharge_vph: float = float(demand_cfg.get("ramp_discharge_vph", ramp_demand_vph))
    ramp_ref_vph: float = ramp_discharge_vph if ramp_model == "metered_queue" else ramp_demand_vph
    veh_len: float = det_cfg["vehicle_length_m"]     # for occupancy fallback

    det_ids = get_detector_ids(config)
    det_ids_per_lane = get_detector_ids_per_lane(config)
    x_grid = get_x_grid(config)
    N_x = len(det_ids)
    if N_x < 3:
        raise ValueError("exit_boundary_flow_vph requires at least 3 detectors")

    t_grid = np.arange(T_ctrl, dtype=np.float32) * dt_ctrl

    # Validate ramp_control shape
    if ramp_control.shape != (T_ctrl,):
        raise ValueError(
            f"ramp_control must have shape ({T_ctrl},), got {ramp_control.shape}"
        )

    density = np.zeros((N_x, T_ctrl), dtype=np.float32)
    speed = np.zeros((N_x, T_ctrl), dtype=np.float32)
    flow = np.zeros((N_x, T_ctrl), dtype=np.float32)

    sumo_cmd = [
        sumo_binary,
        "--net-file", net_file,
        "--route-files", route_file,
        "--additional-files", detector_file,
        "--step-length", str(step_len),
        "--seed", str(seed),
        "--no-step-log",
        "--collision.action", "warn",
    ]
    # Mirror SumoEnv's optional insertion timeout. In metered_queue mode,
    # discarded requests retain their demand in the queue and can be retried.
    max_depart_delay_s = float(sim_cfg.get("max_depart_delay_s", -1.0))
    if max_depart_delay_s >= 0.0:
        sumo_cmd += ["--max-depart-delay", str(max_depart_delay_s)]
    sumo_cmd += [str(a) for a in (sim_cfg.get("sumo_extra_args") or [])]

    veh_counter = 0          # global ramp vehicle ID counter
    frac_accumulator = 0.0   # fractional carry-forward for ramp insertion (open loop)
    meter = (
        MeteredRampQueue(ramp_demand_vph, ramp_discharge_vph, step_len)
        if ramp_model == "metered_queue" else None
    )
    ramp_inflow_vph = np.zeros(T_ctrl, dtype=np.float32)   # vehicles actually inserted
    ramp_queue = np.zeros(T_ctrl, dtype=np.float32)        # virtual queue at end of step
    ramp_departed_count = np.zeros(T_ctrl, dtype=np.int32)
    ramp_pending_count = np.zeros(T_ctrl, dtype=np.int32)
    ramp_pending_ids: set[str] = set()

    # Insertion / teleport counters (logged in metadata)
    total_insert_attempts = 0
    total_insert_success = 0
    total_insert_rejected = 0
    total_ramp_discarded = 0
    total_teleports = 0

    # Ramp queue tracking (vehicles on the ramp edge per sub-step)
    ramp_queue_samples: list[int] = []

    try:
        traci.start(sumo_cmd)

        for k in range(T_ctrl):
            # Per-step accumulators reset each control interval
            sum_count = np.zeros(N_x, dtype=np.float64)
            sum_speed = np.zeros(N_x, dtype=np.float64)
            speed_count = np.zeros(N_x, dtype=np.int32)
            sum_occ = np.zeros(N_x, dtype=np.float64)

            # Rate of ramp vehicle insertion this control step [veh/s] (open loop)
            insert_rate = ramp_control[k] * ramp_demand_vph / 3600.0
            departed_k = 0

            for sub in range(dt_ctrl_steps):
                # --- Ramp vehicle insertion (skipped during warmup) ---
                # warmup_s lets the initial mainline dense wave (caused by
                # departSpeed=max) dissipate before any ramp vehicle arrives
                # at the merge, preventing the cascade-blockage pattern.
                if traci.simulation.getTime() >= warmup_s:
                    if meter is not None:
                        capacity = meter.step(float(ramp_control[k]))
                        # Pending requests still belong to the queue, but
                        # SUMO already owns their IDs. Do not submit them twice.
                        available = max(int(meter.queue) - len(ramp_pending_ids), 0)
                        n_insert = min(capacity, available)
                    else:
                        frac_accumulator += insert_rate * step_len
                        n_insert = int(frac_accumulator)
                        frac_accumulator -= n_insert
                    for _ in range(n_insert):
                        total_insert_attempts += 1
                        veh_id = f"ramp_{veh_counter}"
                        try:
                            traci.vehicle.add(
                                vehID=veh_id,
                                routeID="route_ramp",
                                typeID="passenger",
                                depart=str(traci.simulation.getTime()),
                                departLane="first",
                                departPos="free",   # first collision-free gap on ramp
                                departSpeed="0",
                            )
                            veh_counter += 1
                            total_insert_success += 1
                            ramp_pending_ids.add(veh_id)
                        except traci.exceptions.TraCIException:
                            # A rejected request does not serve any demand.
                            # Metered demand stays queued for a later attempt;
                            # open-loop request capacity is not refunded.
                            total_insert_rejected += 1

                # --- Advance simulation by one step ---
                traci.simulationStep()

                # add() only accepts a request; confirmed departures are the
                # entries onto the road. Count them in the interval they occur,
                # including requests delayed across a control boundary.
                departed_ids = set(traci.simulation.getDepartedIDList())
                pending_ids = set(traci.simulation.getPendingVehicles())
                ramp_departed = ramp_pending_ids & departed_ids
                ramp_discarded = ramp_pending_ids - departed_ids - pending_ids
                ramp_pending_ids -= ramp_departed | ramp_discarded
                departed_k += len(ramp_departed)
                total_ramp_discarded += len(ramp_discarded)
                if meter is not None:
                    meter.on_released(len(ramp_departed))

                # --- Count teleports this step ---
                total_teleports += traci.simulation.getStartingTeleportNumber()

                # --- Sample ramp queue length ---
                ramp_queue_samples.append(
                    traci.edge.getLastStepVehicleNumber("ramp")
                )

                # --- Read detector values for this step ---
                # Aggregate across lanes at each spatial position.
                for j, lane_ids in enumerate(det_ids_per_lane):
                    for det_id in lane_ids:
                        count = traci.inductionloop.getLastStepVehicleNumber(det_id)
                        spd_raw = traci.inductionloop.getLastStepMeanSpeed(det_id)  # m/s or -1
                        occ = traci.inductionloop.getLastStepOccupancy(det_id)      # %

                        sum_count[j] += count
                        sum_occ[j] += occ
                        if spd_raw >= 0.0:
                            sum_speed[j] += spd_raw * count  # count-weighted for averaging
                            speed_count[j] += count

            # --- Aggregate over control interval ---
            flow_vph = sum_count / (dt_ctrl_steps * step_len) * 3600.0  # veh/hr

            mean_speed_mps = np.where(
                speed_count > 0,
                sum_speed / np.maximum(speed_count, 1),
                0.0,
            )
            mean_speed_kmph = mean_speed_mps * 3.6

            # Primary: fundamental relation ρ = q / v
            # Fallback: occupancy-based when speed < 5 km/h
            mean_occ_frac = sum_occ / (dt_ctrl_steps * 100.0)
            density_occ = mean_occ_frac * (1000.0 / veh_len)

            density_fd = np.where(
                mean_speed_kmph > 5.0,
                flow_vph / np.maximum(mean_speed_kmph, 1e-6),
                density_occ,
            )

            density[:, k] = density_fd.astype(np.float32)
            speed[:, k] = mean_speed_kmph.astype(np.float32)
            flow[:, k] = flow_vph.astype(np.float32)
            ramp_departed_count[k] = departed_k
            ramp_pending_count[k] = len(ramp_pending_ids)
            ramp_inflow_vph[k] = departed_k * 3600.0 / dt_ctrl
            ramp_queue[k] = meter.queue if meter is not None else 0.0

    finally:
        traci.close()

    mainline_demand = np.full(T_ctrl, demand_cfg["mainline_demand_vph"], dtype=np.float32)

    return {
        "density": density,
        "speed": speed,
        "flow": flow,
        # Spatial mean at each control timestep, using the interval-averaged
        # detector flows (1700, 1800, 1900 m in the default scenario).
        "exit_boundary_flow_vph": flow[-3:, :].mean(axis=0),
        "x_grid": x_grid,
        "t_grid": t_grid,
        "mainline_demand": mainline_demand,
        # Surrogate branch input: the physical ramp inflow as a fraction of
        # ramp_ref_vph. Open loop: identical to the command (M2 semantics,
        # reference = ramp_demand_vph). Metered queue: inserted flow /
        # ramp_discharge_vph, which can differ from the command due to limited
        # demand or delayed SUMO insertion. The command is kept separately.
        "ramp_control": (
            (ramp_inflow_vph / ramp_ref_vph).astype(np.float32)
            if ramp_model == "metered_queue" else ramp_control.astype(np.float32)
        ),
        "ramp_control_cmd": ramp_control.astype(np.float32),
        "ramp_inflow_vph": ramp_inflow_vph,
        "ramp_queue": ramp_queue,
        "ramp_departed_count": ramp_departed_count,
        "ramp_pending_count": ramp_pending_count,
        "metadata": {
            "seed": seed,
            "demand_profile": demand_cfg["demand_profile"],
            "mainline_demand_vph": demand_cfg["mainline_demand_vph"],
            "ramp_demand_vph": ramp_demand_vph,
            "ramp_model": ramp_model,
            "ramp_discharge_vph": ramp_discharge_vph,
            "ramp_ref_vph": ramp_ref_vph,
            "ramp_flow_measurement": "confirmed_departures",
            "ramp_inflow_max_vph": float(ramp_inflow_vph.max()) if T_ctrl else 0.0,
            "virtual_queue_max": float(ramp_queue.max()) if T_ctrl else 0.0,
            "virtual_queue_final": float(ramp_queue[-1]) if T_ctrl else 0.0,
            "sumo_binary": sumo_binary,
            "T_ctrl": T_ctrl,
            "N_x": N_x,
            # Insertion and teleport diagnostics
            "insert_attempts": total_insert_attempts,
            "insert_success": total_insert_success,
            "insert_rejected": total_insert_rejected,
            "ramp_departed_total": int(ramp_departed_count.sum()),
            "ramp_pending_final": len(ramp_pending_ids),
            "ramp_discarded_total": total_ramp_discarded,
            "teleports": total_teleports,
            "ramp_warmup_s": warmup_s,
            "ramp_queue_max": int(max(ramp_queue_samples)) if ramp_queue_samples else 0,
            "ramp_queue_mean": float(np.mean(ramp_queue_samples)) if ramp_queue_samples else 0.0,
        },
    }
