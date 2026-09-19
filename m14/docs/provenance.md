# Provenance and scope

This folder was extracted on 2026-09-19 from the current working tree of
`traffic-surrogate-rl`, at parent commit `a38da14`. The parent source was left
in place. This is a curated M14 package, not a new scientific experiment.

## Included

- M14 v3b road and demand-family-v2 configuration.
- Frozen validation (18), ID test (30), and OOD test (12) profiles. Profile
  values and SUMO seeds are preserved; the descriptive family path is localized.
- SUMO network construction, queue/insertion bookkeeping, detector aggregation,
  trajectory storage, and episode-level metrics.
- Causal GRU DeepONet, supervised training, bootstrap ensemble, aggregation.
- PPO training in both environments, controller baselines, evaluation and plots.
- Relevant tests, installation metadata, workflow entry point, and guides.

The existing uncommitted queue-statistics fix in `src/sumo_env/rollout.py` was
included. Its copied content was checked byte-for-byte against the original.

## Source mapping

| Original | This folder |
|---|---|
| `configs/sumo/scenario_v3b.yaml` | `configs/scenario.yaml` |
| `configs/profiles/family_v2.yaml` | `configs/demand.yaml` |
| `configs/profiles/v2/*.json` | `configs/profiles/*.json` |
| `configs/experiments/round0_v3b.yaml` | `configs/dataset.yaml` |
| `configs/surrogate/plant_v2.yaml` | `configs/deeponet.yaml` |
| `configs/rl/ppo_common.yaml` + `env_v3b.yaml` | `configs/ppo.yaml` |
| `configs/rl/env_sumo.yaml`, `env_surrogate.yaml` | `configs/env_sumo.yaml`, `env_surrogate.yaml` |
| retained `src/` modules | same relative module names under `src/` |
| retained study scripts | same basenames under `scripts/` |
| shell study orchestration | cross-platform `run.py` |

[source_manifest.json](source_manifest.json) records original and extracted file hashes for the
mapped runtime, scripts, and configurations. It distinguishes direct copies
from adapted files. Hashes describe this extraction, not historical Windows
checkpoint provenance.

## Deliberate cleanup

The package removes earlier density-only DeepONet entry points, padded-MLP and
causal-convolution ablations, one-step surrogates, legacy policy datasets, old
scenario configurations, and milestone scratch scripts. It retains the active
M14 package names to keep imports and saved model compatibility straightforward.

Defaults now point to M14-local paths and configurations. The shared PPO config
includes the v3b overrides explicitly. Physical discharge defaults are 1200
veh/h, while the model's learned inflow normalization remains 1600 veh/h.
Numerical behavior under the explicit M14 configuration is preserved.

Workflow improvements include immediate retry of incomplete ensemble members,
peer-process cleanup on failure, separate ALINEA/constant tuning budgets, and
explicit labeling of accounted runtime rather than elapsed training time.
No model-accuracy or controller-performance improvement is claimed from cleanup.
See [verification.md](verification.md) for the checks performed on this package.

## Historical assets

The historical completed study was recorded as M14 v3b, policy training seed 0,
five rounds with the stop rule firing at round 5. Its initial store was recorded
as 692 trajectories and its ensemble as five models trained for 300 epochs.

The original checkout currently lacks the Windows-generated M14 model and
controller checkpoints, full initial training store/splits, and raw final
controller evaluations. The 404 local E0 trajectories were not silently
promoted to the missing 692-trajectory training store. Large generated files,
unrelated studies, paper files, and virtual environments are not bundled.

Reproduction therefore means generating fresh M14 data and training with this
workflow. Importing historical models later requires their matching ensemble
manifest, normalization, data splits, and controller checkpoints. SUMO 1.27.1
is the locally verified version; the historical Windows SUMO version remains
unconfirmed, so exact cross-machine agreement is not claimed.
