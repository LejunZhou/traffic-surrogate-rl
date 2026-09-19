# Running and extending the study

Run commands from this folder. `run.py` also works when invoked by an absolute
path from another working directory. It anchors child working directories and
imports here, and records measured command elapsed time in `runs/commands.jsonl`.

## Stage inputs and outputs

| Command | Needs | Produces |
|---|---|---|
| `simulate` | frozen profile + chosen controller | `runs/simulation/rollout.npz` and metrics JSON |
| `data` | scenario, demand, dataset mixture | `data/round0/` including splits and normalization |
| `deeponet` | initial dataset | `runs/deeponet/round0/manifest.json`, members, evaluation reports |
| `surrogate-ppo` | ensemble + initial dataset | `runs/aggregation/m14_s0/` |
| `sumo-ppo` | initial density normalization | `runs/study/m14/direct_ppo_*ee_s0/` |
| `baselines` | initial density normalization | `runs/study/m14/alinea_tuning.json` |
| `evaluate` | selected checkpoints + tuned controllers | `runs/study/m14/arms.json` and `eval/*.jsonl` |
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
python run.py sumo-ppo --seed 1 --budgets 200 700
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

For low-level module entry points, set `PYTHONPATH=src` or install this package
in its dedicated environment. The public `run.py` handles this automatically.

Copy the source folder before starting an experiment. Generated manifests and
run records contain resolved absolute paths; moving a trained run requires
updating those paths to its new location.

## Reuse and changing experiments

The public workflow reuses completed initial-data splits, ensemble manifests,
direct-PPO final checkpoints, and baseline reports. Aggregation uses `--resume`.
Incomplete direct-PPO output is rejected instead of restarted into the same
checkpoint directory and ledger. Aggregation can continue completed rounds but
rejects incomplete next-round artifacts; restart those experiments in fresh
output locations. Individual scripts expose more options through `--help`. Reuse is not a cache
key based on every setting: after changing configuration, use a fresh project
copy or explicitly select fresh output paths through the individual scripts.

The full study keeps the M14 300,000 surrogate steps/round and nominal direct
training budgets of 200 and 700 episodes. Validation and selection add simulator
episodes; use the ledger for actual total cost rather than those nominal labels.
For custom direct-PPO budgets, PPO rounds training to complete 480-step rollouts.

Evaluation requests carry a fingerprint of policies, profiles, effective
configuration, and relevant input files. Complete matching results can be
reused. Incomplete or incompatible results fail clearly and require a fresh
output path; existing JSONL files are not blindly accepted or appended to.

The smoke command changes only its isolated copy: 18 constant-controller data
episodes, one-epoch two-member ensemble, one short aggregation round, and short
direct training. Frozen sets are reduced to one profile and one seed each.
The original traffic horizon, road, and network architectures remain intact.

## Baselines and cost

ALINEA and PI-ALINEA selection uses validation profiles. The independent constant
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
