# Milestone 2 plan — Dataset generation (wilson's scenario)

**Status:** in progress. See `_progress/milestone_2_progress.md` for the running log.

## Goal
Generate the supervised training dataset for the DeepONet surrogate
against the current canonical scenario in `configs/sumo/phase1_1.yaml`
(set by wilson's commit `f2bb92f`). Produce raw rollouts + train/val/test
splits in `data/raw/training_data/` and `data/splits/` for M3 to consume.

## Scope (constant-inflow MVP)
Scenario constants pinned by `configs/sumo/phase1_1.yaml`:
- 2000 m mainline, **1 lane**, speed limit 120 km/h.
- On-ramp at **1300 m** with a **100 m acceleration lane** (new edge:
  `highway_accel` with one extra lane between 1300 m and 1400 m).
- Mainline demand **1500 vph** (set by `dataset_constant_inflow.yaml`'s
  `demand_levels: [1500]`, which overrides the 2000 vph default in
  `phase1_1.yaml`), ramp demand **800 vph** (from `phase1_1.yaml`).
- Step length 1 s, control interval 30 s, episode 3600 s → `T_ctrl = 120`.
- **19 detectors** at absolute positions [100, 200, …, 1900] m. Of these,
  12 land on `highway_pre` (x < 1300), 1 on `highway_accel`, 6 on
  `highway_post`.

Dataset config (`configs/experiments/dataset_constant_inflow.yaml`):
- 120 base rollouts, single demand level (2000 vph in the new scenario),
  round-robin over 4 ramp-control families
  (constant / piecewise_constant / smooth / ramp_step).
- Train / val / test split 70 / 15 / 15 on base rollouts.
- Output to **`data/raw/training_data/`** (the new path; the old MVP wrote
  to `data/raw/`, preserved on `mvp-v1-old-scenario`).
- Heatmap PNGs every 10 samples for visual inspection.

Out of scope:
- Time-varying / multi-level mainline demand profiles. Deferred to a
  later M2b once we know whether the surrogate needs more demand
  diversity.
- Truncated / zero-padded control variants. These are now generated at
  *training* time by `TrafficDataset` via the `control_augmentation`
  block in `configs/surrogate/baseline.yaml`, not at dataset-gen time.

## Deliverables
- `data/raw/training_data/sim_0000.npz` … `sim_0119.npz` — 120 rollouts.
- `data/raw/training_data/sim_*_density.png` — sanity heatmaps.
- `data/splits/split_index.json` — train/val/test filename lists.
- `data/splits/metadata.json` — density stats + demand stats + counts.
- `_plans/milestone_2_plan.md` (this file).
- `_progress/milestone_2_progress.md` — running log.

## Sub-milestones
- **2.0 — Smoke run.** `--n-samples 4`. The 1-lane mainline reintroduces
  the teleport risk that Milestone 1.1 first solved with 2 lanes + zipper
  merge. Wilson's geometry (ramp at 1300 m with a 100 m acceleration
  lane) should compensate, but the smoke run is the gate: **0 teleports
  across all 4 samples** is the acceptance condition for scaling to 120.
- **2.1 — Full run.** 120 samples, no overrides. Zero teleports required.
- **2.x — Surface any teleport hot spots in the progress file.** If
  teleports appear at a specific detector or time window, log enough
  detail (which sample, ramp control type, count) that we can decide
  between "tune workaround parameters" and "go back to wilson with the
  scenario change request".

## Acceptance criteria
- Full run completes with **0 total teleports** across all 120 samples.
- All 120 `.npz` files written with the schema documented in
  `src/sumo_env/dataset_generation.py`.
- `data/splits/split_index.json` has 84/18/18 counts (70/15/15 of 120).
- `data/splits/metadata.json` has non-NaN density stats and
  `min_demand == max_demand == 1500.0`.

## Open follow-ups
- **Dataset diagnostics notebook.** A small plot of mean density vs.
  ramp control type, plus a control-type frequency check, would help
  catch dataset issues before M3 training. Not blocking.
- **Higher-demand or multi-level demand.** Defer until M5 results show
  whether the constant-2000-vph dataset is too narrow for PPO.
- The old M2 work for the 1500-vph / 2-lane / 500 m-ramp scenario lives
  on `mvp-v1-old-scenario`; do not re-pull those artifacts.
