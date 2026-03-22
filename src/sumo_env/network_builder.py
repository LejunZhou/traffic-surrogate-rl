"""
Build the SUMO network files programmatically.

Generates for the Phase 1 scenario:
- nodes.nod.xml  : upstream, merge, downstream, ramp_start nodes
- edges.edg.xml  : highway_pre (500 m), highway_post (1500 m), ramp (200 m)
- net.net.xml    : compiled network produced by netconvert
- routes.rou.xml : deterministic passenger vType, routes, and mainline flow

Network topology (all coordinates in metres):
  upstream(0,0) --[highway_pre, 500 m]--> merge(500,0)
                                              |
  ramp_start(327,-100) --[ramp, 200 m]-------+
                                              |
                        --[highway_post, 1500 m]--> downstream(2000,0)

The ramp_start position is chosen so the ramp edge length ≈ 200 m
(30° approach angle: sqrt(173² + 100²) ≈ 200 m).

Junction type at merge is "priority": mainline edges (priority=10) have
right-of-way over the ramp (priority=5), which is standard for ramp metering.

netconvert binary discovery order (see _find_netconvert):
  1. shutil.which("netconvert")      — honours PATH as usual
  2. $SUMO_HOME/bin/netconvert       — common when SUMO_HOME is set but bin/ not on PATH
  3. FileNotFoundError with setup instructions if neither is found
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path


def build_network(output_dir: str, config: dict) -> dict[str, str]:
    """Generate all SUMO network files for the Phase 1 scenario.

    Calls netconvert internally to compile nodes + edges into a .net.xml.

    Args:
        output_dir: Directory where all generated files are written.
        config: Full experiment config dict (uses config["network"],
                config["demand"], config["simulation"]).

    Returns:
        {"net": <abs path to .net.xml>, "route": <abs path to .rou.xml>}

    Raises:
        RuntimeError: If netconvert exits with a non-zero return code.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    nodes_path = od / "nodes.nod.xml"
    edges_path = od / "edges.edg.xml"
    net_path = od / "net.net.xml"
    route_path = od / "routes.rou.xml"

    _write_nodes(nodes_path, config["network"])
    _write_edges(edges_path, config["network"])
    _run_netconvert(nodes_path, edges_path, net_path)
    _write_routes(route_path, config)

    return {
        "net": str(net_path.resolve()),
        "route": str(route_path.resolve()),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _write_nodes(path: Path, net_cfg: dict) -> None:
    ramp_pos = net_cfg["ramp_position_m"]
    ramp_len = net_cfg["ramp_length_m"]
    hw_len = net_cfg["highway_length_m"]

    # Place ramp_start at 30° approach angle so edge length ≈ ramp_len.
    angle = math.radians(30)
    rx = ramp_pos - ramp_len * math.cos(angle)
    ry = -ramp_len * math.sin(angle)

    num_lanes = net_cfg.get("num_lanes", 1)
    # For multi-lane mainline, use "zipper" at merge (cooperative merging)
    # so ramp vehicles don't have to yield to all mainline lanes simultaneously.
    # For single-lane, keep "priority" (original behavior).
    merge_type = "zipper" if num_lanes > 1 else "priority"

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<nodes>\n"
        '    <node id="upstream"   x="0.00"         y="0.00"       type="priority"/>\n'
        f'    <node id="merge"      x="{ramp_pos:.2f}"    y="0.00"       type="{merge_type}"/>\n'
        f'    <node id="downstream" x="{hw_len:.2f}"   y="0.00"       type="priority"/>\n'
        f'    <node id="ramp_start" x="{rx:.2f}"   y="{ry:.2f}" type="priority"/>\n'
        "</nodes>\n"
    )
    path.write_text(content)


def _write_edges(path: Path, net_cfg: dict) -> None:
    spd_main = net_cfg["speed_limit_mps"]
    spd_ramp = net_cfg["ramp_speed_limit_mps"]
    num_lanes = net_cfg.get("num_lanes", 1)

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<edges>\n"
        f'    <edge id="highway_pre"  from="upstream"   to="merge"      '
        f'numLanes="{num_lanes}" speed="{spd_main:.2f}" priority="10"/>\n'
        f'    <edge id="highway_post" from="merge"      to="downstream" '
        f'numLanes="{num_lanes}" speed="{spd_main:.2f}" priority="10"/>\n'
        f'    <edge id="ramp"         from="ramp_start" to="merge"      '
        f'numLanes="1" speed="{spd_ramp:.2f}" priority="5"/>\n'
        "</edges>\n"
    )
    path.write_text(content)


def _find_netconvert() -> str:
    """Locate the netconvert binary, trying PATH then $SUMO_HOME/bin.

    Returns:
        Absolute path string to the netconvert executable.

    Raises:
        FileNotFoundError: If netconvert cannot be found via either method,
            with clear instructions for fixing the setup.
    """
    # 1. Check PATH (covers Homebrew, apt, conda, and manual PATH exports).
    binary = shutil.which("netconvert")
    if binary is not None:
        return binary

    # 2. Try $SUMO_HOME/bin/netconvert (set by the official SUMO installer
    #    and recommended in SUMO docs, but bin/ is not always added to PATH).
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "netconvert"
        if candidate.is_file():
            return str(candidate)

    # 3. Neither found — give an actionable error.
    sumo_home_hint = (
        f"  SUMO_HOME is set to '{sumo_home}' but {Path(sumo_home) / 'bin' / 'netconvert'} "
        "was not found there."
        if sumo_home
        else "  SUMO_HOME is not set."
    )
    raise FileNotFoundError(
        "netconvert not found. To fix this:\n"
        "  • macOS (Homebrew): brew install sumo\n"
        "  • Linux:            sudo apt install sumo  (or download from sumo.dlr.de)\n"
        "  • All platforms:    ensure 'netconvert' is on PATH, or set SUMO_HOME to\n"
        "                      your SUMO installation directory (e.g. /opt/sumo).\n"
        f"{sumo_home_hint}"
    )


def _run_netconvert(nodes: Path, edges: Path, output: Path) -> None:
    """Locate and run SUMO's netconvert tool to compile the network."""
    binary = _find_netconvert()
    cmd = [
        binary,
        "--node-files", str(nodes),
        "--edge-files", str(edges),
        "--output-file", str(output),
        "--no-turnarounds",        # suppress U-turn connections
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert failed (binary: {binary}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _write_routes(path: Path, config: dict) -> None:
    """Write the SUMO routes file with vType, routes, and mainline flow.

    The mainline flow uses a deterministic IDM car-following model
    (sigma=0, speedDev=0) consistent with Phase 1's deterministic setting.

    Ramp vehicles are NOT defined as a flow here; they are inserted
    dynamically via TraCI in run_simulation.py.
    """
    net_cfg = config["network"]
    sim_cfg = config["simulation"]
    demand_cfg = config["demand"]
    veh_cfg = config.get("vehicle", {})

    spd = net_cfg["speed_limit_mps"]
    duration = sim_cfg["duration_s"]
    vph = demand_cfg["mainline_demand_vph"]
    tau = veh_cfg.get("idm_tau_s", 1.0)   # default: SUMO built-in IDM default
    num_lanes = net_cfg.get("num_lanes", 1)
    depart_lane = "best" if num_lanes > 1 else "0"

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<routes>\n"
        '    <!-- Deterministic IDM driver: sigma=0, speedDev=0 (Phase 1). -->\n'
        f'    <!-- IDM headway tau={tau:.2f} s (from config vehicle.idm_tau_s). -->\n'
        '    <vType id="passenger"\n'
        '           carFollowModel="IDM"\n'
        '           accel="2.6"\n'
        '           decel="4.5"\n'
        '           sigma="0.0"\n'
        '           length="5.0"\n'
        '           minGap="2.0"\n'
        f'           tau="{tau:.2f}"\n'
        f'           maxSpeed="{spd:.2f}"\n'
        '           speedFactor="1.0"\n'
        '           speedDev="0.0"/>\n'
        '\n'
        '    <route id="route_main" edges="highway_pre highway_post"/>\n'
        '    <route id="route_ramp" edges="ramp highway_post"/>\n'
        '\n'
        '    <flow id="mainline_flow"\n'
        '          type="passenger"\n'
        '          route="route_main"\n'
        f'          begin="0" end="{duration}"\n'
        f'          vehsPerHour="{vph}"\n'
        f'          departLane="{depart_lane}"\n'
        '          departSpeed="max"/>\n'
        "</routes>\n"
    )
    path.write_text(content)
