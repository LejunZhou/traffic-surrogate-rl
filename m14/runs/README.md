Generated models and evaluations live here:

- `deeponet/round0/`: five trained members and the ensemble manifest.
- `aggregation/m14_s0/`: selected PPO policies, collected trajectories, refined ensembles.
- `study/m14/`: direct PPO runs, baseline tuning, controller manifest, final evaluations.
- `ledger/`: episode-level simulation cost accounting.
- `commands.jsonl`: measured elapsed time and status of each launched command.
- `simulation/`: single-episode demonstrations.
- `smoke/`: isolated short integration runs, including their copied source/configs.

These outputs are ignored by Git. Final evaluation episodes are not training cost.
