# Project: Traffic Flow Control with SUMO, DeepONet, and PPO

## Project goal
This repository studies sample-efficient traffic flow control using:
1. SUMO as the high-fidelity simulator
2. DeepONet as a surrogate dynamics model
3. PPO as the reinforcement learning algorithm for ramp metering control

The core pipeline is:

1. Build a simple traffic simulation in SUMO
2. Generate simulation data under varying inflow/control conditions
3. Train a DeepONet surrogate to map control/input functions to traffic density trajectories
4. Wrap the surrogate as a Gymnasium-compatible environment
5. Train PPO in the surrogate environment
6. Evaluate the learned policy back in SUMO
7. Compare against PPO trained directly in SUMO

## Problem setting
We start with a minimal setting:
- one highway segment
- one direction
- one on-ramp
- fixed road geometry
- controlled ramp metering input
- observable traffic density / speed / flow trajectories

Inputs may include:
- upstream mainline demand profile
- ramp inflow profile or ramp metering signal
- optional initial condition / boundary condition features

Outputs may include:
- density trajectory over time and space
- optional speed / flow trajectory
- cumulative control metrics for RL evaluation

## Research objective
The main objective is to test whether a surrogate model can accelerate RL training while preserving enough fidelity for policy transfer back to the simulator.

Primary comparison:
- PPO trained in surrogate environment
- PPO trained directly in SUMO

Evaluation:
- total reward
- congestion reduction
- throughput
- queue-related metrics
- training wall-clock time
- transfer performance back to SUMO

## Scope control
Start simple. Do NOT add complexity unless explicitly requested.

Phase 1:
- deterministic setting
- single-lane highway (no lane-changing)
- one on-ramp
- baseline DeepONet
- baseline PPO
- evaluation in SUMO

Phase 2 candidates:
- PI-DeepONet
- stochastic demand
- incidents / disturbances
- richer observation models
- more realistic traffic PDE priors
- multi-ramp / multi-agent control

## Phase 1 design decisions

Physical scenario:
- Highway length: 2000 m
- Lanes: 1 (single lane, no lane-changing dynamics)
- On-ramp position: 500 m from upstream boundary
- On-ramp length: 200 m
- Speed limit: 120 km/h (33.33 m/s)
- Simulation duration: 3600 s (1 hour)
- Control step interval: 30 s → T_ctrl = 120 steps
- Detector spacing: 100 m → N_x = 20 detectors along the mainline

Mainline demand:
- Small controlled family of demand profiles, not a single fixed profile
- Profiles: low constant (e.g. 1000 veh/hr), medium constant (1500), high constant (2000), mild peak (ramps from 1200 to 2200 and back)
- At dataset generation time, each simulation samples one demand profile
- At RL episode reset, one demand profile is sampled for the episode

Tooling:
- DeepONet: pure PyTorch (no deepxde)
- PPO: Stable-Baselines3
- Simulation: SUMO with TraCI

Non-goals (Phase 1):
- Multi-ramp or multi-agent control
- Stochastic demand beyond the controlled family
- Incidents or disturbances
- Physics-informed loss in DeepONet
- Lane-changing dynamics
- Real-world calibration

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

## SUMO setup on this Mac

SUMO is installed at `/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO`.

Before running SUMO-based simulations in a fresh shell, set these environment variables:
```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO"
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export PATH="$SUMO_HOME/bin:$PATH"
```

If `sumo`, `netconvert`, or `duarouter` are "not found", the issue is usually environment setup, not a missing installation.

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

## Milestone 2 MVP: dataset generation (2026-03-22)

### Scope
- **Constant mainline demand only**: 1000, 1500, 2000 veh/hr
- **Ramp control families**: constant, piecewise_constant, smooth, ramp_step (equally weighted)
- Time-varying demand (mild_peak) is deferred to Milestone 2b (requires route-generation changes)
- Truncated/zero-padded variants are deferred to Milestone 2b (post-processing; value unclear until RL transfer is tested)
- Surrogate target: density only (speed/flow saved for diagnostics)

### Smoke-test setting
- n_samples=12 (4 per demand level)
- Split: train=8, val=2, test=2
- Config: `configs/experiments/dataset_phase1.yaml` with base SUMO config `configs/sumo/phase1_1.yaml`
- Output tensors per sample: density (20, 120), speed (20, 120), flow (20, 120), x_grid (20,), t_grid (120,), mainline_demand (120,), ramp_control (120,), plus scalar metadata (seed, mainline_demand_vph, ramp_demand_vph)

### Smoke-test result
- 0 teleports across all 12 simulations
- All 3 demand levels covered (1000, 1500, 2000)
- All 4 control types covered
- density shape (20, 120) — correct
- No NaN/Inf in any array
- Split integrity: no overlap between train/val/test
- Normalization stats: mean_density=16.563, std_density=5.745, demand range [1000, 2000]
- Round-trip denormalization check: PASS

### Environment setup and command
Before running, set SUMO environment variables:
```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO"
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export PATH="$SUMO_HOME/bin:$PATH"
```

From the repository root:
```bash
cd src && python -m sumo_env.dataset_generation \
  --config configs/experiments/dataset_phase1.yaml
```

To override sample count for quick tests, add `--n-samples 12`.

### Conclusion
Milestone 2 MVP pipeline is implemented and smoke-tested. Full dataset generation (n=120) and DeepONet surrogate training (Milestone 3) are the next steps. Time-varying demand profiles and truncation/zero-padding variants are deferred follow-up work (Milestone 2b).

## Coding rules
- Python 3.11
- Use type hints
- Prefer small, testable functions
- Keep modules decoupled
- Avoid hardcoded paths
- Use config-driven experiments
- Use clear docstrings for public functions/classes
- Add lightweight tests when practical
- Do not refactor unrelated files unless necessary

## Dependencies
- Python 3.11
- SUMO >= 1.18 (with TraCI, sumolib)
- PyTorch >= 2.0
- Stable-Baselines3 >= 2.0
- Gymnasium >= 0.29
- NumPy, Matplotlib, PyYAML, TensorBoard

## Engineering principles
- Separate simulation, surrogate modeling, and RL code
- Keep dataset schemas stable and documented
- Save all experiments with timestamps or unique run IDs
- Never overwrite prior results silently
- Prefer reproducibility over cleverness
- Make the minimal working version first

## Repository structure target
traffic-surrogate-rl/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── sumo/
│   ├── surrogate/
│   ├── rl/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── sumo_env/
│   │   ├── network_builder.py
│   │   ├── run_simulation.py
│   │   ├── detectors.py
│   │   └── dataset_generation.py
│   ├── surrogate/
│   │   ├── deeponet.py
│   │   ├── datasets.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── rl/
│   │   ├── surrogate_env.py
│   │   ├── sumo_env_wrapper.py
│   │   ├── reward.py
│   │   ├── train_ppo.py
│   │   └── evaluate.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── plotting.py
├── scripts/
│   ├── make_dataset.sh
│   ├── train_surrogate.sh
│   ├── train_ppo_surrogate.sh
│   └── eval_in_sumo.sh
└── notebooks/

## Dataset conventions
All generated datasets should clearly specify:
- input representation
- target representation
- time grid
- space grid
- normalization assumptions
- train/val/test split

Preferred saved fields:
- mainline demand
- ramp control input
- density trajectory (supervised target for DeepONet)
- speed trajectory (logged for diagnostics, not a training target in Phase 1)
- flow trajectory (logged for diagnostics, not a training target in Phase 1)
- metadata (seed, sim settings, network settings)

## DeepONet I/O contract (Phase 1)

Architecture: unstacked DeepONet with dot-product output.

Branch net input:
- Concatenation of [ramp_control(t); mainline_demand(t)]
- Shape: (2 * T_ctrl,) = (240,)
- ramp_control values ∈ [0, 1] (metering rate)
- mainline_demand values normalized (min-max across the demand family)

Trunk net input:
- Query coordinates (x, t), each normalized to [0, 1]
- x normalized by highway length L; t normalized by simulation duration T_sim
- Shape per query point: (2,)
- During training, query points are sampled from the full (N_x × T_ctrl) grid

Output:
- Predicted density ρ(x, t) at each query point
- Shape: (N_query,)
- Density is z-score normalized (mean/std computed from training set)

Supervised target (Phase 1):
- **Density only.** Speed and flow are collected from SUMO for logging and evaluation diagnostics but are NOT used as surrogate training targets in Phase 1.

Training pipeline:
- MSE loss on density predictions
- Batched training with checkpointing and validation
- Evaluation: predicted-vs-true density heatmaps, per-sample L2 error

For future extensions:
- Physics-informed loss (PDE residual) — Phase 2
- Speed/flow as additional targets — Phase 2
- Do not bake PI logic into baseline unless requested

## Surrogate rollout strategy (Phase 1)

The DeepONet is trained on complete input functions: given a full ramp control profile u(t) and demand profile d(t), it predicts the density field ρ(x, t) at any query point.

During RL, the control signal is constructed incrementally — one action per step. At RL step k (0-indexed):
1. The agent has chosen actions u(0), u(1), ..., u(k)
2. The branch input is constructed as: [u(0), ..., u(k), 0, ..., 0 ; d(0), ..., d(T-1)]
   - Ramp control: first k+1 entries are actual actions, remainder zero-padded to T_ctrl
   - Demand: always the full profile (known in advance for the episode)
3. The trunk queries all detector positions at time t_k: {(x_i, t_k) for i = 1..N_x}
4. The DeepONet returns density predictions at those points → this becomes the observation
5. The reward is computed from this density snapshot

This means the DeepONet is re-evaluated from scratch at every RL step (not autoregressive).

Known risk — distribution shift:
Training data contains fully-specified control signals. During RL rollout, partially-specified (zero-padded) signals are a distribution shift. The surrogate may produce unreliable density predictions for the zero-padded future portion, but we only query density at the current time t_k (not future times), which partially mitigates this.

Phase 1 dataset design requirement:
To support the zero-padded rollout formulation, the training dataset MUST include truncated/zero-padded control variants:
- For each full simulation trajectory, generate additional training samples by truncating the control signal at random cut points k ∈ {1, ..., T_ctrl-1} and zero-padding the remainder
- Query points for truncated samples should be restricted to t ≤ t_k (only the valid portion)
- This is not optional — it is a core requirement for the surrogate to generalize to RL rollout conditions

Additional mitigations:
1. Monitor surrogate prediction error during RL evaluation by comparing surrogate predictions against SUMO ground truth for the same control sequence
2. If transfer gap is large, consider autoregressive reformulation in Phase 2

## RL interface contract (Phase 1)

Observation design:
- Shape: (N_x + 2,) = (22,)
- Components:
  - density[0:N_x]: density at each of the 20 detector locations at current time step (z-score normalized, same normalization as surrogate training)
  - demand[N_x]: current mainline demand at this time step (min-max normalized to [0, 1])
  - time[N_x+1]: current normalized time index = k / T_ctrl ∈ [0, 1]
- Justification: the agent needs the current traffic state (density field) to decide the metering rate, the current demand level to adapt across demand regimes, and the time index because optimal metering strategy is time-dependent (e.g. more aggressive metering early when congestion is building vs. releasing the queue near episode end). Without time, the policy is forced to be purely reactive; with time, it can anticipate demand evolution. We do NOT include speed/flow (redundant given density in single-lane), ramp queue length (not directly available from the surrogate in Phase 1), or past actions (the surrogate handles temporal dependence internally via the full control history in the branch input).

Action:
- Shape: (1,)
- Semantics: ramp metering rate ∈ [0, 1]
  - 0.0 = ramp fully closed (no vehicles enter from on-ramp)
  - 1.0 = ramp fully open (all ramp demand enters)
- Continuous action space (Box)

Reward (Phase 1 baseline):
- Computed by a shared reward function in src/rl/reward.py, used identically by both surrogate and SUMO environments
- Phase 1 baseline: reward = -(mean density across all N_x detectors at current step)
- Lower density → higher reward → less congestion
- Must operate on the same scale in both environments (denormalize surrogate predictions before reward computation, or compute reward in normalized space consistently)
- Future reward extensions (Phase 2+): add throughput bonus, on-ramp queue length penalty, and/or total travel time terms. The baseline reward is intentionally simple to validate the pipeline before adding multi-objective shaping.

Episode structure:
- Episode length: T_ctrl = 120 steps (one full simulation horizon = 3600 s)
- At reset: sample a demand profile from the controlled family
- No early termination in Phase 1

Environment parity:
- SurrogateEnv and SumoEnv must expose identical observation_space, action_space, and reward function
- PPO training code must be fully agnostic to which env it uses

## Experiment rules
Every experiment should:
- read from config files
- save metrics and plots
- save seeds and hyperparameters
- produce a compact summary at the end

Minimum outputs:
- training curves
- evaluation metrics
- predicted-vs-true plots for surrogate
- PPO reward curves
- final SUMO evaluation table

## What to optimize for
Prioritize:
1. correctness
2. reproducibility
3. clear interfaces
4. modularity
5. only then performance

## When making changes
When implementing a task:
- first inspect relevant files
- propose a small plan internally
- change only files needed for the task
- run lightweight validation if possible
- summarize what changed and any assumptions

## Current milestone order
1. Minimal SUMO network + rollout script
2. Dataset generation pipeline
3. Baseline DeepONet training
4. Gymnasium-compatible surrogate environment
5. PPO training
6. Evaluation in SUMO
7. Comparison study and plots