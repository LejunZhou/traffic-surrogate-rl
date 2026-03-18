"""
Launch a single SUMO simulation via TraCI and collect trajectory data.

Each call to run_simulation:
1. Starts SUMO with the given network + route + detector configuration
2. Steps through the simulation, applying the ramp metering signal at each control step
3. Reads detector outputs at each control step
4. Returns structured trajectory arrays

Ramp metering control:
- ramp_control[k] ∈ [0, 1] is applied at control step k
- 0.0 = ramp fully closed; 1.0 = ramp fully open
- Implemented via a SUMO/TraCI-compatible ramp metering control mechanism
  to be finalized in Milestone 1
"""

from __future__ import annotations

import numpy as np

# TODO: implement SUMO/TraCI simulation runner


def run_simulation(
    net_file: str,
    route_file: str,
    detector_file: str,
    ramp_control: np.ndarray,
    config: dict,
) -> dict:
    """Run one SUMO simulation episode and return trajectory data.

    Args:
        net_file: Path to the .net.xml file.
        route_file: Path to the .rou.xml file.
        detector_file: Path to the .add.xml detector file.
        ramp_control: Shape (T_ctrl,). Metering rate at each control step, ∈ [0, 1].
        config: Simulation config (dt_ctrl, T_ctrl, seed, sumo_binary, etc.).

    Returns:
        Dict with keys:
            "density": np.ndarray, shape (N_x, T_ctrl), veh/km
            "speed":   np.ndarray, shape (N_x, T_ctrl), km/h
            "flow":    np.ndarray, shape (N_x, T_ctrl), veh/hr
            "metadata": dict (seed, demand profile id, sim config)
    """
    raise NotImplementedError
