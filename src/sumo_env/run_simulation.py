"""
Launch a single SUMO simulation via TraCI and collect trajectory data.

Each call to run_simulation:
1. Starts SUMO (headless) with the given network / route / detector files.
2. Steps through the simulation in 1 s increments.
3. Every dt_ctrl_s seconds (one control step), applies the ramp metering
   signal and reads all 20 induction-loop detectors.
4. Returns structured numpy arrays matching the Phase 1 dataset schema.

Ramp metering control mechanism (finalised in Milestone 1):
  At each 1 s SUMO step a fractional-accumulator decides whether to insert
  a ramp vehicle via traci.vehicle.add().  The accumulation rate is:

      rate [veh/s] = ramp_control[k] × ramp_demand_vph / 3600

  This gives approximately ramp_control[k] × ramp_demand_vph × dt_ctrl / 3600
  vehicles per control step, distributed evenly across the 30 sub-steps.
  A carry-forward fractional accumulator prevents systematic under/over-insertion.

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

from sumo_env.detectors import get_detector_ids, get_x_grid


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
            "x_grid":          np.float32, shape (N_x,),        metres from upstream
            "t_grid":          np.float32, shape (T_ctrl,),     seconds (start of each interval)
            "mainline_demand": np.float32, shape (T_ctrl,),     veh/hr (constant for this profile)
            "ramp_control":    np.float32, shape (T_ctrl,),     the input ramp_control array
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
    veh_len: float = det_cfg["vehicle_length_m"]     # for occupancy fallback

    det_ids = get_detector_ids(config)
    x_grid = get_x_grid(config)
    N_x = len(det_ids)

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

    veh_counter = 0          # global ramp vehicle ID counter
    frac_accumulator = 0.0   # fractional carry-forward for ramp insertion

    # Insertion / teleport counters (logged in metadata)
    total_insert_attempts = 0
    total_insert_success = 0
    total_insert_rejected = 0
    total_teleports = 0

    try:
        traci.start(sumo_cmd)

        for k in range(T_ctrl):
            # Per-step accumulators reset each control interval
            sum_count = np.zeros(N_x, dtype=np.float64)
            sum_speed = np.zeros(N_x, dtype=np.float64)
            speed_count = np.zeros(N_x, dtype=np.int32)
            sum_occ = np.zeros(N_x, dtype=np.float64)

            # Rate of ramp vehicle insertion this control step [veh/s]
            insert_rate = ramp_control[k] * ramp_demand_vph / 3600.0

            for sub in range(dt_ctrl_steps):
                # --- Ramp vehicle insertion (skipped during warmup) ---
                # warmup_s lets the initial mainline dense wave (caused by
                # departSpeed=max) dissipate before any ramp vehicle arrives
                # at the merge, preventing the cascade-blockage pattern.
                if traci.simulation.getTime() >= warmup_s:
                    frac_accumulator += insert_rate * step_len
                    n_insert = int(frac_accumulator)
                    frac_accumulator -= n_insert
                    for _ in range(n_insert):
                        total_insert_attempts += 1
                        try:
                            traci.vehicle.add(
                                vehID=f"ramp_{veh_counter}",
                                routeID="route_ramp",
                                typeID="passenger",
                                depart=str(traci.simulation.getTime()),
                                departLane="first",
                                departPos="free",   # first collision-free gap on ramp
                                departSpeed="0",
                            )
                            veh_counter += 1
                            total_insert_success += 1
                        except traci.exceptions.TraCIException:
                            # SUMO rejected insertion (e.g. no space on ramp).
                            # The vehicle is skipped; the fractional accumulator
                            # does NOT refund it to avoid repeated retry loops.
                            total_insert_rejected += 1

                # --- Advance simulation by one step ---
                traci.simulationStep()

                # --- Count teleports this step ---
                total_teleports += traci.simulation.getStartingTeleportNumber()

                # --- Read detector values for this step ---
                for j, det_id in enumerate(det_ids):
                    count = traci.inductionloop.getLastStepVehicleNumber(det_id)
                    spd_raw = traci.inductionloop.getLastStepMeanSpeed(det_id)  # m/s or -1
                    occ = traci.inductionloop.getLastStepOccupancy(det_id)      # %

                    sum_count[j] += count
                    sum_occ[j] += occ
                    if spd_raw >= 0.0:
                        sum_speed[j] += spd_raw
                        speed_count[j] += 1

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

    finally:
        traci.close()

    mainline_demand = np.full(T_ctrl, demand_cfg["mainline_demand_vph"], dtype=np.float32)

    return {
        "density": density,
        "speed": speed,
        "flow": flow,
        "x_grid": x_grid,
        "t_grid": t_grid,
        "mainline_demand": mainline_demand,
        "ramp_control": ramp_control.astype(np.float32),
        "metadata": {
            "seed": seed,
            "demand_profile": demand_cfg["demand_profile"],
            "mainline_demand_vph": demand_cfg["mainline_demand_vph"],
            "ramp_demand_vph": ramp_demand_vph,
            "sumo_binary": sumo_binary,
            "T_ctrl": T_ctrl,
            "N_x": N_x,
            # Insertion and teleport diagnostics
            "insert_attempts": total_insert_attempts,
            "insert_success": total_insert_success,
            "insert_rejected": total_insert_rejected,
            "teleports": total_teleports,
            "ramp_warmup_s": warmup_s,
        },
    }
