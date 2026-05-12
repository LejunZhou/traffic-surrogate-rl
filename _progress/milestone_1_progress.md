# Milestone 1 progress

Retrospective notes on Milestone 1 (minimal SUMO network + rollout) and
Milestone 1.1 (two-lane zipper merge fix). Moved here from proposal.md.
See `_plans/milestone_1_plan.md` for the original plan.

## Milestone 1.1 results: two-lane mainline with zipper merge

Key conclusion: the decisive fix for ramp-merge teleports was **2 mainline lanes + zipper (cooperative) merge junction**. Lane count alone was not sufficient — a 2-lane priority junction produced the same teleport failure pattern as the 1-lane baseline because ramp vehicles still had to yield to all conflicting mainline lanes simultaneously.

Working baseline for future ramp-merge experiments: 2-lane mainline + zipper junction at the merge node.

Observed results (all using phase1_1.yaml workaround parameters):

| Config | Ramp rate | Teleports | Inserts | Mean density | Mean flow |
|--------|-----------|-----------|---------|-------------|-----------|
| 1-lane priority | 0.5 | 11 | 289/289 | 12.43 | 1368 |
| 2-lane priority | 0.5 | 11 | 289/289 | 11.47 | 1293 |
| 2-lane zipper | 0.5 | 0 | 289/289 | 14.16 | 1594 |
| 2-lane zipper | 1.0 | 0 | 580/580 | 16.90 | 1856 |

The higher density/flow in zipper runs is expected: ramp vehicles enter the network instead of being teleported away.

Status: Milestone 1 workaround parameters (ramp_warmup_s=120, idm_tau_s=1.5, mainline_demand_vph=1200, ramp_length_m=500, ramp_speed_limit_mps=27.78) remain in place. Parameter rollback ablation is a separate follow-up task.

## Milestone 1.1 rerun (2026-03-22)

Command:
```bash
python scripts/run_rollout.py --config configs/sumo/phase1_1.yaml --ramp-rate 0.5 --output-index rerun_1_1
```

Results (using the current `phase1_1.yaml` configuration at the time of this rerun):
- Teleports: 0
- Inserts: 299/299, 0 rejected
- Mean density: 17.36 veh/km
- Mean flow: 1987 veh/hr
- Ramp queue: max=2, mean=1.2

These metrics differ from the earlier Milestone 1.1 table above because the config has since been updated (e.g. `mainline_demand_vph` changed from 1200 to 1500, workaround parameters removed). Zero teleports confirms the zipper merge continues to work correctly.
