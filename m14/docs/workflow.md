# Running and extending the study

Run commands from this folder. `run.py` also works when invoked by an absolute
path from another working directory. It anchors child working directories and
imports here, and records measured command elapsed time in `runs/commands.jsonl`.

## Stage inputs and outputs

| Command | Needs | Produces |
|---|---|---|
| `simulate` | frozen profile + chosen controller | `runs/simulation/rollout.npz` and metrics JSON |
| `e0` | scenario, demand | `runs/study/m14/e0.json`, `e0_capacity_comparison.json` (vs `configs/reference/e0_ramp120kmh.json`) |
| `data` | scenario, demand, dataset mixture | `data/round0/` including splits and normalization |
| `deeponet` | initial dataset | `runs/deeponet/round0/manifest.json`, members, evaluation reports |
| `surrogate-ppo` | ensemble + initial dataset | `runs/aggregation/m14_s0/` |
| `sumo-ppo` | initial density normalization | `runs/study/m14/direct_ppo_*ee_s0/` |
| `baselines` | initial density normalization | `runs/study/m14/alinea_tuning.json` |
| `evaluate` | selected checkpoints + tuned controllers | `runs/study/m14/arms.json` and `eval/*.jsonl` (ALINEA, PI-ALINEA and Surrogate-MPC on the last aggregation ensemble included) |
| `tables` | evaluations + last aggregation ensemble | `runs/study/m14/tables/` (Table I, Table II, headline TTS reductions; summed compute hours) |
| `report` | manifest + evaluations | `reports/figures/` |

`data` uses 404 E0 scenario-screening trajectories and 288 mixture trajectories.
The mixture is 25% wide ALINEA, 15% dithered ALINEA, 25% store-and-flush,
15% feedforward, 10% random signal, and 10% constant control. Dataset splitting
and normalization follow the original M14 implementation. Model-validation
trajectories are separate from the frozen controller-validation demand set.

The DeepONet evaluation stage writes gate diagnostics on both held-out splits.
As in the historical driver, these diagnostics are reported rather than used
as an automatic pipeline stop. Inspect them before interpreting policy results.

## Commands you may want

```bash
# Inspect a profile without training.
python run.py simulate --set ood --index 2 --policy u=1 --seed 101

# Limit CPU use and inspect the full command plan.
python run.py pipeline --workers 4 --dry-run

# Another policy training seed, using the same initial ensemble/data.
python run.py surrogate-ppo --seed 1 --rounds 5 --workers 4
python run.py sumo-ppo --seed 1 --budgets 1000
python run.py evaluate --seeds 0 1 --workers 4

# Evaluate chosen policies without constructing the full study manifest.
python scripts/eval_policy_profiles_sumo.py \
  --policies runs/aggregation/m14_s0/A1_final.zip u=0.5 \
  --set test --workers 4 --out runs/eval/custom_test.jsonl

# Plot density and exit-flow predictions.
python scripts/plot_plant_eval.py \
  --ensemble runs/deeponet/round0 --split test --out reports/deeponet
```

`A1_final.zip` is the policy with the best SUMO validation score across rounds.
The best round may be earlier than 5, and the stop rule may end training early.
Selection details are recorded in `runs/aggregation/m14_s0/study.json`.
`pipeline --seed 1` trains and evaluates seed 1; it does not launch a seed sweep.
`evaluate --seeds 0 1` includes those already-trained seeds.
`seeds --new-seeds 1 2` is the seed sweep for a finished seed-0 study: every new
seed's surrogate-PPO loop and direct PPO run as concurrent branches (logs
`runs/logs/pipeline_{surrogate,direct}_s<seed>.log`), then `evaluate`, `tables`
and `report` cover seed 0 and the new seeds. Seed 0's evaluations are reused (same
request fingerprints). `tables --seeds 0 1 2` writes each seed's tables to
`tables/seed_<s>/` and a summary to `tables/tables.md`: mean ± sd over seeds and
headline reductions with a hierarchical bootstrap CI (seeds, then episodes).
`extend-direct --from-budget 1000 --to-budget 1200` trains the finished direct PPO
runs of every seed (`--seeds`, default 0 1 2) longer: `direct_ppo_1200ee_s<seed>`
starts as a copy of the 1000-episode run whose final model (policy, value net,
optimizer, step count) is its latest checkpoint, with its validation history and
ledger, so the longer run is charged for all its episodes and selects its
checkpoint over the whole run. The runs continue concurrently, then `evaluate`,
`tables` and `report` cover both budgets: Table II's SUMO-PPO row is the larger
budget and `SUMO-PPO (1000 ep.)` keeps the smaller one, with headline reductions
against both. A resumed session's checkpoints and evaluations stay on the
`eval_freq` grid even when it starts from an off-grid final model.

For low-level module entry points, set `PYTHONPATH=src` or install this package
in its dedicated environment. The public `run.py` handles this automatically.

Copy the source folder before starting an experiment. Generated manifests and
run records contain resolved absolute paths; moving a trained run requires
updating those paths to its new location.

## Reuse and changing experiments

The public workflow reuses completed initial-data splits, ensemble manifests,
direct-PPO final checkpoints, and baseline reports. Aggregation uses `--resume`.
An interrupted direct-PPO run continues from its latest checkpoint
(`training.resume`): optimizer state and step count are restored, the
evaluation history and the ledger are cut back to that checkpoint (episodes
simulated after it move to `ledger_lost.jsonl` in the run directory, logs of
earlier sessions are kept as `*_part<n>`), and the profile / SUMO-seed streams
continue instead of replaying. A resumed run is not bitwise identical to an
uninterrupted one. Aggregation can continue completed rounds but by default
rejects incomplete next-round artifacts. With `pipeline --recover-interrupted`
(or `M14_RECOVER_INTERRUPTED=1`), an interrupted aggregation round and an
interrupted evaluation request are moved aside (`interrupted_*`, never deleted)
and redone; their SUMO episodes are not charged, so reported cost is that of an
uninterrupted run. Use it only when no other study process is running.
Individual scripts expose more options through `--help`. Reuse is not a cache
key based on every setting: after changing configuration, use a fresh project
copy or explicitly select fresh output paths through the individual scripts.

The surrogate environment advances each member's GRU by one input per control
step (`BranchCache`) instead of rerunning it over the whole history; outputs are
identical up to float round-off and PPO steps run about 5-8x faster
(`env.incremental_branch: false` restores the full recompute). `--torch-threads N`
caps the CPU threads of every PPO process (`M14_TORCH_THREADS`); without it torch
takes all cores, which oversubscribes the CPU when several studies share one
machine. 2-4 threads per process were fastest in a local benchmark.

The full study keeps the M14 300,000 surrogate steps/round; the direct-PPO
default is one run at a nominal 1000 episodes, validated every 9600 steps
(4800 for budgets up to 200). Validation and selection add simulator
episodes; use the ledger for actual total cost rather than those nominal labels.
For custom direct-PPO budgets, PPO rounds training to complete 480-step rollouts.
`pipeline` stops before generating data when the E0 capacity check recommends
rescaling the capacity-tied constants (override: `--ignore-capacity-check`).

Evaluation requests carry a fingerprint of policies, profiles, effective
configuration, and relevant input files. Complete matching results can be
reused. Incomplete or incompatible results fail clearly and require a fresh
output path; existing JSONL files are not blindly accepted or appended to.

The smoke command changes only its isolated copy: 18 constant-controller data
episodes, one-epoch two-member ensemble, one short aggregation round, and short
direct training. Frozen sets are reduced to one profile and one seed each.
The original traffic horizon, road, and network architectures remain intact.

## Baselines and cost

ALINEA and PI-ALINEA selection uses validation profiles. The default search covers
detector stations 12-15 (1300-1600 m), set-points 20-34 veh/km, integral gains
10/20/35 and PI gains 2/4/8 (540 episodes; the original M14 grid had 324 and its
chosen detector and set-point were both grid-edge values). `alinea_tuning.json`
records the grid and an `edge_check`; a selected value on the edge of its grid
means the grid should be widened in that direction and the tuning repeated.
ALINEA is charged its own candidates, PI-ALINEA the whole joint search. The independent constant
sweep evaluates 11 rates; its cost is not charged again to ALINEA. Fixed
references u=0,0.5,1 have zero training/tuning cost. The tuned constant is a
separate manifest entry even if it happens to select u=0.5.

Surrogate-MPC is available through `mpc:<ensemble-directory>` policy specs and
the manifest builder's explicit `--mpc-spec`. It is optional in the full M14
workflow. Its demand access and online compute differ from the reactive PPO
policy; disclose these when comparing it.

Manifest `accounted_runtime_s` is partial accounting of simulator and learning
work, not measured end-to-end elapsed training time. Use `runs/commands.jsonl`
for measured stage elapsed time; nested or concurrently running stages must
not be blindly summed. The historical comparison figures report return.
For TTS comparisons, use episode-level `tts_veh_h`, matching profile and SUMO seed.
