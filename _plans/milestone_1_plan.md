# Milestone 1 plan — Minimal SUMO network + rollout script

**Status:** done. See `_progress/milestone_1_progress.md` for results.

## Goal
Stand up the smallest viable SUMO simulation of the Phase 1 physical scenario
and expose a single-rollout entry point, so that later milestones (dataset
generation, surrogate, RL) have a working simulator to build on.

## Scope
Phase 1 physical scenario (from proposal.md):
- Highway length 2000 m, on-ramp at 500 m, ramp length 200 m
- Speed limit 120 km/h, simulation duration 3600 s
- Control step interval 30 s → T_ctrl = 120 steps
- Detector spacing 100 m → N_x = 20 detectors
- Deterministic, 2-lane mainline with zipper merge at the on-ramp (decided permanent after Milestone 1.1)

## Deliverables
- `src/sumo_env/network_builder.py` — programmatic network construction via netconvert
- `src/sumo_env/run_simulation.py` — single-rollout runner using TraCI
- `src/sumo_env/detectors.py` — 20 mainline detectors at 100 m spacing
- `scripts/run_rollout.py` — CLI entry point to run one rollout with a given config and ramp rate
- `configs/sumo/phase1.yaml` — baseline SUMO config for the Phase 1 scenario

## Sub-milestones
- **1.0** — Minimal single-lane pipeline (network builder, run_simulation, detectors, run_rollout.py) producing density/speed/flow time series at the detectors.
- **1.1** — Fix ramp-merge teleports. Resolution: two-lane mainline + zipper (cooperative) merge junction. Lane count alone was insufficient; the zipper junction is the decisive fix. Working baseline config: `configs/sumo/phase1_1.yaml`.

## Acceptance criteria
- `scripts/run_rollout.py` runs end-to-end against `configs/sumo/phase1_1.yaml` without errors.
- Zero teleports on the baseline ramp-rate sweep.
- All ramp vehicles successfully inserted (no rejected inserts).
- Detector output covers the full 3600 s horizon with expected shape.

## Open follow-ups
- **Parameter rollback ablation** — the Milestone 1 workaround parameters
  (ramp_warmup_s, idm_tau_s, mainline_demand_vph, ramp_length_m,
  ramp_speed_limit_mps) remain in place in `configs/sumo/phase1_1.yaml`.
  A clean ablation to roll each back and confirm the zipper junction alone is
  sufficient is deferred; not currently scheduled.
