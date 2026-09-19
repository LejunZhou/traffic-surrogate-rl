# Workflow scripts

Use `python run.py <command>` from the `m14/` folder for the supported workflow. These scripts expose lower-level controls and are adapted from the same-named files in the parent repository's `scripts/` directory. Their defaults refer only to this standalone folder.

| Script | Purpose | Public command |
| --- | --- | --- |
| `run_scenario_characterisation.py` | Check insertion and compare fixed/dynamic metering schedules; save E0 trajectories. | `data` |
| `generate_round0_dataset.py` | Generate the six-family M14 behavior mixture and train/validation/test split. | `data` |
| `train_ensemble.py` | Train or fine-tune bootstrap DeepONet members and write their manifest. | `deeponet`; internally `surrogate-ppo` |
| `eval_surrogate_regimes.py` | Measure density, exit-flow, return, and congestion-regime errors. | `deeponet` |
| `run_aggregation_loop.py` | Train surrogate PPO, select checkpoints in SUMO, collect trajectories, and fine-tune DeepONet. | `surrogate-ppo` |
| `select_checkpoint_profiles.py` | Select direct-SUMO PPO checkpoints using saved validation returns. | `sumo-ppo` |
| `tune_alinea_profiles.py` | Tune ALINEA/PI-ALINEA and constant metering on validation demand. | `baselines` |
| `build_arms_manifest.py` | Gather controller paths, episode budgets, and evaluation destinations. | `evaluate` |
| `run_final_evaluation.py` | Evaluate manifest controllers on held-out test/OOD demand. | `evaluate` |
| `eval_policy_profiles_sumo.py` | Evaluate explicit controller specifications on a selected profile set. | Advanced direct script; `simulate` handles one episode. |
| `plot_sample_efficiency.py` | Plot controller returns, episode budgets, transfer gaps, and OOD performance. | `report` |
| `plot_plant_eval.py` | Plot held-out SUMO/DeepONet fields, flow, and prediction errors after model evaluation. | Advanced direct script after `deeponet`. |

`pipeline` runs the public stages in order. `smoke` exercises a reduced workflow in a separate copy. Each script supports `--help`.

The manifest's `accounted_runtime_s` is partial accounting of recorded SUMO episode and learning-process durations, not elapsed wall-clock time or normalized compute. Measured public-command elapsed times are recorded in `runs/commands.jsonl`.

Aggregation `--resume` continues only at completed round boundaries. It preserves recorded rounds and starts the next round when that round has no existing training, rollout, evaluation, store, or ledger artifacts. If a round was interrupted, the command stops and preserves all files for inspection; restart with a new study name and output directory instead of silently deleting partial work. Relocating generated aggregation outputs requires rebasing their saved absolute paths.

Final evaluations validate the requested policies, demand profiles, and environment against the saved request before reusing results. Existing output files are passed to that validation; their existence alone does not mean evaluation completed.

The OOD and breakdown comparison figures use the A1 round selected by SUMO validation for each policy seed, and the largest available requested direct-PPO budget for each seed. They do not choose controllers using held-out outcomes. Cost curves retain all recorded rounds and budgets.
