"""
Place E1 (induction loop) detectors along the mainline highway and
provide helpers for reading them via TraCI.

Detector layout (Phase 1, N_x=20, spacing=100 m):
  Detectors 00–04 : edge "highway_pre"  at pos  50, 150, 250, 350, 450 m
  Detectors 05–19 : edge "highway_post" at pos  50, 150, …, 1450 m

Absolute positions from the upstream boundary (x_grid):
  50, 150, 250, …, 1950 m  (100 m spacing, 20 values)

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

    n: int = det_cfg["n_detectors"]            # 20
    spacing: float = det_cfg["spacing_m"]      # 100.0
    freq: int = sim_cfg["dt_ctrl_s"]           # 30
    ramp_pos: float = net_cfg["ramp_position_m"]  # 500.0

    n_pre = int(ramp_pos / spacing)            # 5  (on highway_pre)
    n_post = n - n_pre                         # 15 (on highway_post)

    # SUMO requires a file attribute even when we read via TraCI.
    det_out = Path(output_path).parent / "det_output.xml"

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]

    for i in range(n_pre):
        pos = (i + 0.5) * spacing
        lines.append(
            f'    <inductionLoop id="det_{i:02d}" '
            f'lane="highway_pre_0" '
            f'pos="{pos:.1f}" '
            f'freq="{freq}" '
            f'file="{det_out}"/>'
        )

    for i in range(n_post):
        pos = (i + 0.5) * spacing
        idx = n_pre + i
        lines.append(
            f'    <inductionLoop id="det_{idx:02d}" '
            f'lane="highway_post_0" '
            f'pos="{pos:.1f}" '
            f'freq="{freq}" '
            f'file="{det_out}"/>'
        )

    lines.append("</additional>")
    Path(output_path).write_text("\n".join(lines))
    return str(Path(output_path).resolve())


def get_detector_ids(config: dict) -> list[str]:
    """Return ordered detector IDs from upstream to downstream.

    Args:
        config: Full experiment config dict.

    Returns:
        List of strings ["det_00", "det_01", …, "det_19"].
    """
    n: int = config["detectors"]["n_detectors"]
    return [f"det_{i:02d}" for i in range(n)]


def get_x_grid(config: dict) -> np.ndarray:
    """Return detector absolute positions in metres from the upstream boundary.

    Args:
        config: Full experiment config dict.

    Returns:
        shape (N_x,) float32 array: [50., 150., …, 1950.]
    """
    n: int = config["detectors"]["n_detectors"]
    spacing: float = config["detectors"]["spacing_m"]
    return np.array([(i + 0.5) * spacing for i in range(n)], dtype=np.float32)


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
