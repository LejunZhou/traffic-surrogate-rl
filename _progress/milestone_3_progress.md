# Milestone 3 progress

Running log for the new M3 (baseline DeepONet training on wilson's
scenario, using `surrogate/train.py` + `control_augmentation`). See
`_plans/milestone_3_plan.md` for scope and acceptance.

## 2026-05-11 — Kickoff

- Plan file written: `_plans/milestone_3_plan.md`.
- Todo list refreshed (7 tasks for M3): plan → progress skeleton →
  wandb-off → smoke → full → eval → docs commit.
- Dependencies in place:
  - M2 dataset committed (`e6a5db4`): 120 rollouts in
    `data/raw/training_data/`, splits 84/18/18 in `data/splits/`,
    density stats mean=18.730, std=5.971.
  - Shaped reward already committed (`b021df7`); M3 doesn't use reward
    directly but M4/M5 will.

## Config preparation

- `configs/surrogate/baseline.yaml` edits:
  - `data.constant_mainline_demand_vph: 1500` (explicit, was null).
    Required because `apply_sumo_config_defaults` auto-fills the null
    from `phase1_1.yaml`'s `mainline_demand_vph: 2000`, but our M2
    dataset is at 1500. Without this fix, `TrafficDataset` would
    filter out every sample.
  - `output.wandb.enabled: false` (was true). Avoids the credential
    prompt for the first run; TensorBoard logs are still produced.
- New file `configs/surrogate/baseline_smoke.yaml` — same as
  baseline.yaml but `n_epochs=5`, `eval_every=1`, `run_name=deeponet_smoke`.
  Lets the smoke and full configs stay isolated.

## 3.0 — Smoke run

Command: `python -m surrogate.train --config configs/surrogate/baseline_smoke.yaml`

Run dir: `runs/surrogate/deeponet_smoke_20260511_234741/`

Key facts confirmed:
- **Train views: 1428** (`full=84`, `padded=1344`) — augmentation
  produces 17 views per rollout exactly as planned (1 full + 16 padded).
  Val views: 18 (full only); test would be the same.
- Density stats loaded from training set: `mean=18.7296`, `std=5.9709`,
  matching M2's `metadata.json`.
- `apply_sumo_config_defaults` succeeded — `branch_input_dim=auto` was
  resolved without error.
- Per-epoch wall time ≈ 10 s (5 epochs in well under a minute on CPU).

Per-epoch metrics:

| epoch | train_mse | val_mse |
|---|---|---|
| 1 | 0.9993 | 0.6932 |
| 2 | 0.9324 | 0.6050 |
| 3 | 0.7704 | 0.4888 |
| 4 | 0.7092 | **0.4632** |
| 5 | 0.6618 | 0.6062 |

Observations:
- Convergence is **much faster** than the old M3 smoke (which only
  reached val_mse 0.74 in 5 epochs on the un-augmented 84-sample
  set). Same architecture; the difference is the richer training data.
- Val MSE bouncing 0.46 → 0.61 between epochs 4 and 5 is normal noise
  for a small val set (18 rollouts).
- Best checkpoint saved at epoch 4, val_mse=0.4632.

Decision: full 1000-epoch run as configured. Estimated ~3 hours of CPU
time based on the smoke per-epoch (10s × 1000 ≈ 167 min). Will be
interruptible because `best.pt` saves continuously.

## 3.1 — Full training

Command: `python -m surrogate.train --config configs/surrogate/baseline.yaml`
(background; output mirrored to `_progress/m3_full_run.log`).

Run dir: `runs/surrogate/deeponet_constant_inflow_20260511_234849/`

Highlights:
- 1000 epochs completed. (Wall time was longer than the smoke
  extrapolation suggested — ran overnight rather than ~3 hours.)
- Final `train_mse ≈ 0.049`, `val_mse ≈ 0.063`.
- **Best `val_mse = 0.0591`** — about 5.7× better than the old MVP's
  0.340 on the un-augmented dataset, holding everything else equal.
  The 17×-larger training set (84 full + 1344 padded views) is doing
  the bulk of the work.
- Train and val curves stayed close throughout (no overfitting).

Artifacts in run dir:
- `best.pt`, `final.pt`, `config.yaml`, `normalization.json`,
  `metrics.csv` (1000 rows).
- TensorBoard scalars under `runs/surrogate/.../` (default SB3 location).

## 3.2 — Test-set evaluation

Command: `python -m surrogate.eval --checkpoint runs/surrogate/deeponet_constant_inflow_20260511_234849/best.pt`

Result:
```json
{
  "mean_l2": 67.58,
  "mean_rel_l2": 0.0738,
  "mean_mse_physical": 2.13,
  "n_samples": 18
}
```

- `mean_rel_l2 = 0.0738` — **well below the 0.10 "comfortable for RL"
  target** and far below the 0.15 acceptance gate. Old MVP was 0.112.
- Per-detector RMSE ≈ 1.46 veh/km in physical units (down from old's
  1.92 veh/km).
- Heatmaps saved: `eval/sample_000.png`, `sample_001.png`, `sample_002.png`.

## Acceptance verdict

PASS (comfortably).

- Smoke run completed under 1 minute.
- Full 1000-epoch run completed; train + val curves converged together,
  no divergence.
- Test rel-L2 = 0.0738 ≤ 0.15 (and < 0.10 too).
- 3 sample heatmaps render.

## Checkpoint pointer (for M4)

**Absolute path:**
`C:\Users\Jun18\Desktop\CS 285\traffic-surrogate-rl\runs\surrogate\deeponet_constant_inflow_20260511_234849\best.pt`

**Repo-relative:**
`runs/surrogate/deeponet_constant_inflow_20260511_234849/best.pt`

`SurrogateEnv` loads this via `torch.load(map_location="cpu",
weights_only=False)` and reads `model_state_dict`, `config`,
`normalization` keys. Normalization stats match M2 metadata
(mean=18.730, std=5.971).
