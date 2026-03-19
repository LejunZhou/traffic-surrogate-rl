"""
Run all four Milestone 1 diagnostic scenarios in sequence and print a
comparison table.

Usage (from project root):
    python scripts/run_diagnostic_suite.py
    python scripts/run_diagnostic_suite.py --dry-run
    python scripts/run_diagnostic_suite.py --scenarios free_flow,stress

Scenarios
---------
1. free_flow  configs/sumo/diag_free_flow.yaml   ramp-rate=0.0
   Very low mainline (500 veh/hr), ramp closed.
   Validates fundamental-diagram extraction under pure free-flow conditions.

2. no_ramp    configs/sumo/phase1.yaml            ramp-rate=0.0
   Medium mainline (1200 veh/hr), ramp closed.
   Validates mainline-only flow with no merge pressure.

3. moderate   configs/sumo/phase1.yaml            ramp-rate=0.5
   Medium mainline (1200 veh/hr), half ramp rate (300 effective veh/hr).
   Phase 1 default scenario; mild merge bottleneck expected.

4. stress     configs/sumo/diag_stress.yaml       ramp-rate=1.0
   Near-capacity mainline (1800 veh/hr) + full ramp (600 veh/hr) = 2400 veh/hr.
   Deliberately over-saturates the merge; severe congestion expected.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUN_ROLLOUT = _PROJECT_ROOT / "scripts" / "run_rollout.py"

# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS: list[dict] = [
    {
        "name": "free_flow",
        "config": "configs/sumo/diag_free_flow.yaml",
        "ramp_rate": 0.0,
        "output_index": "diag_free_flow",
        "description": "500 veh/hr mainline, ramp closed — pure free-flow baseline",
    },
    {
        "name": "no_ramp",
        "config": "configs/sumo/phase1.yaml",
        "ramp_rate": 0.0,
        "output_index": "diag_no_ramp",
        "description": "1200 veh/hr mainline, ramp closed — mainline-only validation",
    },
    {
        "name": "moderate",
        "config": "configs/sumo/phase1.yaml",
        "ramp_rate": 0.5,
        "output_index": "diag_moderate",
        "description": "1200 veh/hr mainline, ramp rate=0.5 — Phase 1 default",
    },
    {
        "name": "stress",
        "config": "configs/sumo/diag_stress.yaml",
        "ramp_rate": 1.0,
        "output_index": "diag_stress",
        "description": "1800 veh/hr mainline, ramp rate=1.0 — overloaded merge",
    },
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all Milestone 1 diagnostic scenarios and compare results"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them",
    )
    p.add_argument(
        "--scenarios",
        default="",
        metavar="NAME,...",
        help=(
            "Comma-separated subset of scenario names to run "
            "(default: all four). "
            "Valid names: free_flow, no_ramp, moderate, stress"
        ),
    )
    return p.parse_args()


# ── Summary parsing ───────────────────────────────────────────────────────────

def _parse_summary(stdout: str) -> dict:
    """Extract numeric values from the printed summary block."""
    result: dict = {}

    for line in stdout.splitlines():
        m = re.search(r"density\s*:.*mean=([0-9.]+).*max=([0-9.]+)", line)
        if m:
            result["density_mean"] = float(m.group(1))
            result["density_max"] = float(m.group(2))

        m = re.search(r"speed\s*:.*mean=([0-9.]+)", line)
        if m:
            result["speed_mean"] = float(m.group(1))

        m = re.search(r"flow\s*:.*mean=([0-9.]+)", line)
        if m:
            result["flow_mean"] = float(m.group(1))

        m = re.search(
            r"insertions:\s*(\d+)/(\d+)\s*ok,\s*(\d+)\s*rejected", line
        )
        if m:
            result["insert_ok"] = int(m.group(1))
            result["insert_total"] = int(m.group(2))
            result["insert_rejected"] = int(m.group(3))

        m = re.search(r"teleports\s*:\s*(\d+)", line)
        if m:
            result["teleports"] = int(m.group(1))

    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, dry_run: bool) -> tuple[bool, dict, float]:
    """Run one scenario; return (success, parsed_stats, elapsed_s)."""
    cmd = [
        sys.executable,
        str(_RUN_ROLLOUT),
        "--config", scenario["config"],
        "--ramp-rate", str(scenario["ramp_rate"]),
        "--output-index", scenario["output_index"],
    ]

    print(f"\n{'─' * 70}")
    print(f"  Scenario : {scenario['name']}")
    print(f"  Config   : {scenario['config']}")
    print(f"  Ramp rate: {scenario['ramp_rate']}")
    print(f"  Command  : {' '.join(cmd[2:])}")   # omit python path for brevity
    print(f"{'─' * 70}")

    if dry_run:
        print("  (dry-run: skipped)")
        return True, {}, 0.0

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=False,    # stream stdout/stderr to terminal in real time
        text=True,
    )
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"\n  [FAILED] Exit code {proc.returncode}")
        return False, {}, elapsed

    # Re-run just to capture stdout for parsing (the run already completed above)
    # Instead: run again silently to parse? No — parse from a second captured run
    # is wasteful. We stream live above and do a silent capture for stats only.
    # The trade-off: we run SUMO once (live) and once more silently (stats only).
    # BETTER: run once, capture output, AND print it.
    # Refactor: run with capture, tee to stdout manually.
    return True, {}, elapsed


def run_scenario_with_capture(
    scenario: dict, dry_run: bool
) -> tuple[bool, dict, float]:
    """Run one scenario, stream output to terminal, and parse summary stats."""
    cmd = [
        sys.executable,
        str(_RUN_ROLLOUT),
        "--config", scenario["config"],
        "--ramp-rate", str(scenario["ramp_rate"]),
        "--output-index", scenario["output_index"],
    ]

    print(f"\n{'─' * 70}")
    print(f"  Scenario : {scenario['name']}")
    print(f"  Desc     : {scenario['description']}")
    print(f"  Config   : {scenario['config']}")
    print(f"  Ramp rate: {scenario['ramp_rate']}")
    print(f"{'─' * 70}\n")

    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return True, {}, 0.0

    t0 = time.monotonic()

    # Capture output (so we can parse it) while also printing it live.
    # subprocess.Popen lets us do both.
    import io
    captured = io.StringIO()
    with subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.write(line)
        proc.wait()

    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"\n  [FAILED] returncode={proc.returncode}")
        return False, {}, elapsed

    stats = _parse_summary(captured.getvalue())
    return True, stats, elapsed


# ── Comparison table ──────────────────────────────────────────────────────────

def _fmt(val: object, fmt: str = ".1f") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return format(val, fmt)
    return str(val)


def print_table(results: list[dict]) -> None:
    """Print a compact side-by-side comparison of all scenario results."""
    header = (
        f"{'scenario':<14}  {'d_mean':>7}  {'d_max':>7}  "
        f"{'spd_mean':>8}  {'flow_mean':>9}  "
        f"{'inserts':>8}  {'teleports':>9}  {'time_s':>7}  status"
    )
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  DIAGNOSTIC SUITE COMPARISON")
    print(f"{'═' * len(header)}")
    print(f"  {header}")
    print(f"  {sep}")

    for r in results:
        s = r.get("stats", {})
        dm = _fmt(s.get("density_mean"), ".2f")
        dx = _fmt(s.get("density_max"), ".2f")
        sp = _fmt(s.get("speed_mean"), ".1f")
        fl = _fmt(s.get("flow_mean"), ".0f")
        ok = s.get("insert_ok")
        tot = s.get("insert_total")
        ins = f"{ok}/{tot}" if ok is not None else "—"
        tp = _fmt(s.get("teleports", None), "d") if s.get("teleports") is not None else "—"
        el = _fmt(r.get("elapsed", 0.0), ".0f")
        status = "OK" if r.get("success") else "FAILED"

        print(
            f"  {r['name']:<14}  {dm:>7}  {dx:>7}  "
            f"{sp:>8}  {fl:>9}  "
            f"{ins:>8}  {tp:>9}  {el:>7}  {status}"
        )

    print(f"  {sep}")
    print(f"  density [veh/km]  speed [km/h]  flow [veh/hr]\n")


# ── Expected-behaviour reference ──────────────────────────────────────────────

def print_expected() -> None:
    """Print the qualitative expected behaviour for each scenario."""
    rows = [
        (
            "free_flow",
            "≈ 4.2 flat",
            "≈ 120 flat",
            "0",
            "none",
            "Density = q/v = 500/120. Perfectly flat across space and time.",
        ),
        (
            "no_ramp",
            "≈ 10 flat",
            "≈ 120 flat",
            "0",
            "none",
            "Density = 1200/120 = 10. Flat — ramp is closed, no merge.",
        ),
        (
            "moderate",
            "step at x=500",
            "slight drop",
            "0–few",
            "mild step",
            "Pre-merge ≈10, post-merge ≈12.5 veh/km (1200+300)/120. Small density jump.",
        ),
        (
            "stress",
            "peak near x=500",
            "drop to <30",
            "possible",
            "severe",
            "Over-capacity merge (2400 > 2100 veh/hr). Queue spills back along highway_pre.",
        ),
    ]

    print(f"\n{'─' * 95}")
    print("  EXPECTED BEHAVIOUR REFERENCE")
    print(f"{'─' * 95}")
    hdr = f"  {'scenario':<14}  {'density':>16}  {'speed':>12}  {'teleports':>9}  {'bottleneck':>9}  notes"
    print(hdr)
    print(f"  {'─' * 93}")
    for name, den, spd, tp, bn, notes in rows:
        print(f"  {name:<14}  {den:>16}  {spd:>12}  {tp:>9}  {bn:>9}")
        print(f"  {'':>14}  {notes}")
    print(f"{'─' * 95}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Filter scenarios
    selected_names = (
        {n.strip() for n in args.scenarios.split(",") if n.strip()}
        if args.scenarios
        else {s["name"] for s in SCENARIOS}
    )
    valid_names = {s["name"] for s in SCENARIOS}
    unknown = selected_names - valid_names
    if unknown:
        print(f"Unknown scenario names: {unknown}", file=sys.stderr)
        print(f"Valid names: {sorted(valid_names)}", file=sys.stderr)
        sys.exit(1)

    selected = [s for s in SCENARIOS if s["name"] in selected_names]

    print_expected()

    if args.dry_run:
        print("(dry-run mode — no SUMO calls will be made)\n")

    # Run scenarios
    all_results: list[dict] = []
    for scenario in selected:
        success, stats, elapsed = run_scenario_with_capture(scenario, args.dry_run)
        all_results.append(
            {
                "name": scenario["name"],
                "success": success,
                "stats": stats,
                "elapsed": elapsed,
            }
        )

    # Comparison table
    if not args.dry_run:
        print_table(all_results)

    # Exit non-zero if any scenario failed
    if any(not r["success"] for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
