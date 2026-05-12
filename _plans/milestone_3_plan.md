# Milestone 3 plan — Baseline DeepONet training (wilson's pipeline)

**Status:** in progress. See `_progress/milestone_3_progress.md` for the running log.

## Goal
Train the baseline DeepONet surrogate against the M2 dataset
(`data/raw/training_data/`, wilson's `phase1_1.yaml` scenario), using
wilson's restructured `src/surrogate/train.py`. Produce a checkpoint + test
metrics that meet the rel-L2 ≤ 0.15 acceptance gate, ready for M4 to load.

## Scope
What wilson's pipeline does that the old M3 didn't:
- `apply_sumo_config_defaults()` auto-derives `highway_length_m`,
  `duration_s`, `dt_ctrl_s`, `n_detectors`, and
  `constant_mainline_demand_vph` from `data.sumo_config = configs/sumo/phase1_1.yaml`.
- `branch_input_dim: "auto"` → `T_ctrl = 3600 / 30 = 120`.
- **Control augmentation in `TrafficDataset`**: per train rollout, 1 full
  view + 16 padded prefix views (random prefix lengths in `[1, 119]`).
  With 84 train samples this gives ~84 × 17 = 1428 effective training
  views per epoch — much richer than the old M3's 84.

Architecture (`baseline.yaml`):
- Unstacked DeepONet, branch_input_dim=120, trunk_input_dim=2,
  hidden=128, latent=128, GELU.
- MSE loss on z-score-normalized density (training set stats from M2:
  mean=18.730, std=5.971).
- AdamW, lr=1e-3, weight_decay=1e-6, grad clip 1.0.
- `n_query_points=512` random query points per training view; val/test
  use the full 19×120 grid (padded views use the smaller in-causal grid).
- 1000 epochs, eval every 10 epochs. wandb disabled for first run.

Out of scope:
- 240-dim branch (concatenated demand). Single-demand MVP, so demand is
  filtered to 1500 vph and not fed as a separate branch input.
- Multi-demand training. Deferred to M2b/M3b if M5 results require it.

## Deliverables
- `runs/surrogate/deeponet_constant_inflow_<timestamp>/best.pt` — best
  checkpoint by val MSE.
- `runs/surrogate/.../final.pt` — last-epoch checkpoint.
- `runs/surrogate/.../config.yaml`, `normalization.json`, `metrics.csv`.
- `runs/surrogate/.../eval/eval_metrics.json` + 3 sample heatmaps.
- `_plans/milestone_3_plan.md` (this file).
- `_progress/milestone_3_progress.md` — running log.

## Sub-milestones
- **3.0 — Smoke run.** 5 epochs, `eval_every=1`. Confirm the new
  augmentation kicks in (train views ≈ 17 × 84 = 1428), config flows
  through `apply_sumo_config_defaults`, checkpoint saves.
- **3.1 — Full training.** 1000 epochs. Watch for divergence /
  pathological loss curves; expect higher per-epoch wall time than the
  old M3 because of the ~17× larger view count.
- **3.2 — Test-set evaluation.** `surrogate.eval` against `best.pt`.
  Acceptance gate: `mean_rel_l2 ≤ 0.15`. Also generate 3 sample heatmaps.

## Acceptance criteria
- Smoke run completes without errors.
- Full run completes; train + val MSE curves both non-divergent.
- `eval/eval_metrics.json` reports `mean_rel_l2 ≤ 0.15`.
- 3 sample heatmaps render and predicted closely tracks true.
- The absolute path to the chosen checkpoint is recorded in the progress
  file for M4 to read.

## Open follow-ups
- **M3b — variable-demand branch + 240-dim input.** Required only if M5
  RL transfer suggests the single-demand surrogate is too narrow.
- **Eval-mode augmentation.** Currently val/test use only the full
  control view. We could optionally also evaluate on padded prefix views
  to surface any partial-control prediction errors before M4. Not
  blocking.
- **Hyperparameter sweep.** Defaults pass acceptance on the old MVP;
  expected to also pass here, but record val MSE plateau in the progress
  file in case a deeper net or longer training would help.
