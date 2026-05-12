# Milestone 2 progress

Running log for the Milestone 2 dataset rerun against wilson's
`phase1_1.yaml` scenario. See `_plans/milestone_2_plan.md` for scope
and acceptance.

## 2026-05-11 — Kickoff

- Plan file written: `_plans/milestone_2_plan.md`.
- Todo list refreshed (5 new tasks for the rerun): plan → progress
  skeleton → smoke → full → docs commit.
- Pre-flight check:
  - `git status` clean; on `main` ahead by 2 (UTF-8 fix + shaped reward).
  - `configs/sumo/phase1_1.yaml`: 1-lane, ramp@1300m, accel lane 100m,
    2000 vph / 800 vph, 19 detectors at [100..1900] m.
  - `configs/experiments/dataset_constant_inflow.yaml`: writes to
    `data/raw/training_data/`. 120 samples, 4 control types,
    splits 70/15/15.
- The shaped-reward overhaul is independent of M2 (M2 doesn't call
  reward.py). M2 can run while it's committed.

## 2.0 — Smoke run (4 samples)

Command:
`python -m sumo_env.dataset_generation --config configs/experiments/dataset_constant_inflow.yaml --n-samples 4`

Output:
- `data/raw/training_data/` directory created (new path; old MVP data
  still at `data/raw/` and stays untouched).
- Network built once, routes built for 1500 vph (note: the dataset
  config's `demand_levels: [1500]` overrides `phase1_1.yaml`'s 2000 vph
  default; intentional — gives a milder regime than full SUMO+RL).
- `[1/4] demand=1500, ctrl=constant, inserts=619/619, OK`
- `[2/4] demand=1500, ctrl=piecewise_constant, inserts=555/555, OK`
- `[3/4] demand=1500, ctrl=smooth, inserts=471/471, OK`
- `[4/4] demand=1500, ctrl=ramp_step, inserts=376/376, OK`
- **`All simulations had 0 teleports.`** — the 1-lane mainline +
  acceleration lane geometry behaves correctly at this demand. The
  teleport concern from the plan does not materialize.
- `splits: train=2 val=1 test=1, density mean=18.933 std=5.938,
  demand min=1500 max=1500`.

Notes:
- Insert counts (~370–620) are well above zero and below the demand
  cap, so neither floor nor ceiling effects are active.
- Density std (5.94) is noticeably higher than the prior MVP's 3.30 on
  the old scenario, consistent with the longer pre-merge section
  (1300 m vs. 500 m before) creating more spatial variation.

Smoke acceptance: PASS. Scaling to 120.

## 2.1 — Full run (120 samples)

Command:
`python -m sumo_env.dataset_generation --config configs/experiments/dataset_constant_inflow.yaml --overwrite`

(`--overwrite` so the 4 smoke files at sim_0000..sim_0003 are replaced
in place — same seeds + same control types means the regenerated files
are bit-identical.)

Highlights:
- 120 simulations, **0 teleports total**, no rejected inserts.
- Insert counts ranged from ~234 to ~752 vehicles per simulation
  depending on the sampled ramp control profile.
- Splits: `train=84, val=18, test=18` (70/15/15 of 120).
- Training-set stats: density `mean=18.730 veh/km`, `std=5.971 veh/km`.
  Density mean is similar to the smoke-run subset (18.93 → 18.73 with
  more samples). Std is much higher than the old MVP's 3.30 — the
  ramp at 1300 m creates a downstream congestion zone that doesn't
  appear in the upstream half, increasing spatial variance.
- Demand `min == max == 1500.0` (constant by design).

## Artifact verification

- `data/raw/training_data/sim_0000.npz` … `sim_0119.npz` — 120 files
  present. Per-sample shape check on `sim_0050.npz`:
  - keys: `['density', 'flow', 'mainline_demand', 'mainline_demand_vph',
    'ramp_control', 'ramp_demand_vph', 'seed', 'speed', 't_grid', 'x_grid']`
  - `density.shape == (19, 120)` float32 — **N_x = 19** as expected.
  - `ramp_control.shape == (120,)` — T_ctrl = 120.
  - `x_grid == [100, 200, 300, …, 1900]` — matches the new
    `detectors.py` formula `(start_position_m=100) + i * (spacing=100)`.
  - `mainline_demand_vph == 1500.0`.
- 12 heatmap PNGs in `data/raw/training_data/`.
- `data/splits/split_index.json` — train/val/test lists pairwise
  disjoint (verified empty intersections).
- `data/splits/metadata.json`: `mean_density=18.730`, `std_density=5.971`,
  `min_demand=1500.0`, `max_demand=1500.0`, `n_total=120`, `seed=42`.

## Acceptance verdict

PASS.

- 120 / 120 samples generated.
- 0 teleports across the run.
- Split cardinalities and metadata fields match the plan.
- Sample schema matches `(N_x=19, T_ctrl=120)` from wilson's new
  geometry; ready for M3.

## Notes for downstream milestones

- **Higher density std** (5.97 vs prior MVP's 3.30) means more
  trajectory diversity for the surrogate to learn. Should help fitting
  rather than hurt. Heatmaps for first few samples (saved at indices
  0, 10, 20, …, 110) confirm visible congestion downstream of the
  1300 m ramp.
- **Single demand level** is unchanged from the prior MVP. With the
  shaped reward we've already committed, the policy now has the queue
  + density-std counterbalance that the old setup lacked. If the
  surrogate-PPO eval still shows degenerate behavior in M5, the next
  follow-up is a multi-demand dataset (M2b) and a 240-dim branch
  input on the DeepONet (M3b).
