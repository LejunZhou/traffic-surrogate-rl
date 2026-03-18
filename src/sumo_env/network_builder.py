"""
Build the SUMO network files programmatically.

Generates:
- .net.xml : single-lane highway (2000 m) with one on-ramp (at 500 m, length 200 m)
- .rou.xml : vehicle routes and demand flows for a given demand profile
- .sumocfg : simulation configuration tying together network, routes, and detectors

Phase 1 scenario (from CLAUDE.md):
- Highway length: 2000 m, 1 lane
- On-ramp position: 500 m from upstream boundary, length 200 m
- Speed limit: 120 km/h (33.33 m/s)
- Simulation duration: 3600 s
"""

# TODO: implement network_builder


def build_network(output_dir: str, config: dict) -> dict[str, str]:
    """Generate SUMO network files for the Phase 1 scenario.

    Args:
        output_dir: Directory where .net.xml, .rou.xml, .sumocfg are written.
        config: Network configuration (highway length, on-ramp position, etc.).

    Returns:
        Mapping of file type to absolute path:
        {"net": ..., "route": ..., "cfg": ...}
    """
    raise NotImplementedError
