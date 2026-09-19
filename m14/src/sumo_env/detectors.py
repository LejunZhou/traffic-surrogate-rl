"""
Place E1 (induction loop) detectors along the mainline highway and
provide helpers for reading them via TraCI.

Detector layout (Phase 1, spacing=100 m, start_position_m=200 m):
  Detectors are placed at absolute positions 200, 300, ..., 1900 m
  for the default 2000 m highway.

Absolute positions from the upstream boundary (x_grid):
  start_position_m + i * spacing_m.

If network.acceleration_lane_length_m is set, detectors whose absolute
position falls between the merge and the lane drop are placed on edge
"highway_accel", which has one extra lane for ramp acceleration.

Detectors are read online via TraCI (getLastStepVehicleNumber,
getLastStepMeanSpeed, getLastStepOccupancy); the XML output file
written by SUMO is not parsed in Phase 1.

Note: parse_detector_output (XML-file parsing) is kept as a stub for
future offline use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_detector_file(output_path: str, config: dict) -> str:
    """Generate the SUMO additional file (.add.xml) with E1 induction loops.

    One loop is placed at the centre of each 100 m spacing interval.
    The freq attribute controls XML file aggregation (set to dt_ctrl_s);
    TraCI online reads are per-step regardless of freq.

    Args:
        output_path: Path to write the .add.xml file.
        config: Full experiment config dict.

    Returns:
        Absolute path to the written .add.xml file.
    """
    det_cfg = config["detectors"]
    sim_cfg = config["simulation"]
    net_cfg = config["network"]

    n: int = det_cfg["n_detectors"]
    freq: int = sim_cfg["dt_ctrl_s"]           # 30
    # SUMO requires a file attribute even when we read via TraCI.
    det_out = Path(output_path).parent / "det_output.xml"

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]

    for i in range(n):
        edge_id, edge_pos, lane_count = _detector_edge_at_index(config, i)
        for lane in range(lane_count):
            det_id = _detector_id(i, lane_count, lane)
            lines.append(
                f'    <inductionLoop id="{det_id}" '
                f'lane="{edge_id}_{lane}" '
                f'pos="{edge_pos:.1f}" '
                f'freq="{freq}" '
                f'file="{det_out}"/>'
            )

    lines.append("</additional>")
    Path(output_path).write_text("\n".join(lines))
    return str(Path(output_path).resolve())


def get_detector_ids(config: dict) -> list[str]:
    """Return ordered spatial detector IDs from upstream to downstream.

    For single-lane configs, returns ["det_00", …, "det_19"].
    For multi-lane configs, also returns ["det_00", …, "det_19"] (spatial only).
    Use get_detector_ids_per_lane() to get per-lane IDs for aggregation.

    Args:
        config: Full experiment config dict.

    Returns:
        List of N_x spatial ID strings.
    """
    n: int = config["detectors"]["n_detectors"]
    return [f"det_{i:02d}" for i in range(n)]


def get_detector_ids_per_lane(config: dict) -> list[list[str]]:
    """Return per-lane detector IDs grouped by spatial position.

    Args:
        config: Full experiment config dict.

    Returns:
        List of N_x lists, each containing num_lanes detector ID strings.
        Single-lane: [["det_00"], ["det_01"], …]
        Multi-lane:  [["det_00_L0", "det_00_L1"], ["det_01_L0", "det_01_L1"], …]
    """
    n: int = config["detectors"]["n_detectors"]
    result: list[list[str]] = []
    for i in range(n):
        _, _, lane_count = _detector_edge_at_index(config, i)
        result.append([_detector_id(i, lane_count, lane) for lane in range(lane_count)])
    return result


def get_x_grid(config: dict) -> np.ndarray:
    """Return detector absolute positions in metres from the upstream boundary.

    Args:
        config: Full experiment config dict.

    Returns:
        shape (N_x,) float32 array of detector absolute positions.
    """
    return np.array(_detector_positions(config), dtype=np.float32)


def parse_detector_output(output_xml: str, config: dict) -> dict:
    """Parse SUMO detector output XML into numpy arrays.

    Stub — retained for future offline (non-TraCI) dataset generation.

    Args:
        output_xml: Path to the detector output file written by SUMO.
        config: Full experiment config dict.

    Returns:
        Dict with keys "density", "speed", "flow", each shape (N_x, T_ctrl).
    """
    raise NotImplementedError(
        "parse_detector_output is reserved for offline XML parsing (Milestone 2+). "
        "Phase 1 reads detectors online via TraCI in run_simulation.py."
    )


def _detector_id(index: int, lane_count: int, lane: int) -> str:
    return f"det_{index:02d}" if lane_count == 1 else f"det_{index:02d}_L{lane}"


def _detector_edge_at_index(config: dict, index: int) -> tuple[str, float, int]:
    """Return edge id, edge-local detector position, and lane count."""
    x_abs = _detector_positions(config)[index]
    return _detector_edge_at_position(config, x_abs)


def _detector_positions(config: dict) -> list[float]:
    det_cfg = config["detectors"]
    net_cfg = config["network"]

    n = int(det_cfg["n_detectors"])
    spacing = float(det_cfg["spacing_m"])
    start = float(det_cfg.get("start_position_m", 2.0 * spacing))
    highway_length = float(net_cfg["highway_length_m"])
    positions = [start + i * spacing for i in range(n)]
    if positions and positions[-1] >= highway_length:
        raise ValueError(
            "Detector grid extends beyond the highway. "
            f"Last detector position is {positions[-1]:.1f} m, "
            f"but highway_length_m is {highway_length:.1f} m. "
            "Reduce detectors.n_detectors or detectors.start_position_m."
        )
    return positions


def _detector_edge_at_position(config: dict, x_abs: float) -> tuple[str, float, int]:
    net_cfg = config["network"]

    ramp_pos = float(net_cfg["ramp_position_m"])
    num_lanes = int(net_cfg.get("num_lanes", 1))
    accel_len = float(net_cfg.get("acceleration_lane_length_m", 0.0))

    if x_abs < ramp_pos:
        return "highway_pre", x_abs, num_lanes

    if accel_len > 0.0:
        accel_end = ramp_pos + accel_len
        if x_abs < accel_end:
            return "highway_accel", x_abs - ramp_pos, num_lanes + 1
        return "highway_post", x_abs - accel_end, num_lanes

    return "highway_post", x_abs - ramp_pos, num_lanes
