"""
Define and place SUMO detectors along the mainline highway.

Uses SUMO lane-area detectors (E2) placed at regular spatial intervals.
Detectors measure density, speed, and flow aggregated over each control step.

Phase 1 parameters (from CLAUDE.md):
- N_x = 20 detectors
- Detector spacing: 100 m
- Aggregation interval: 30 s (matches control step dt_ctrl)
"""

# TODO: implement detector placement and output parsing


def build_detector_file(output_path: str, config: dict) -> str:
    """Generate the SUMO additional file (.add.xml) placing E2 detectors.

    Args:
        output_path: Path to write the .add.xml file.
        config: Detector config (spacing, edge ids, aggregation interval).

    Returns:
        Absolute path to the written .add.xml file.
    """
    raise NotImplementedError


def parse_detector_output(output_xml: str, config: dict) -> dict:
    """Parse SUMO detector output XML into numpy arrays.

    Args:
        output_xml: Path to the detector output file written by SUMO.
        config: Detector config (N_x, T_ctrl, etc.).

    Returns:
        Dict with keys "density", "speed", "flow", each shape (N_x, T_ctrl).
    """
    raise NotImplementedError
