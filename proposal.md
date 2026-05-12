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
- 1-lane mainline highway + 100 m acceleration lane + single-lane on-ramp
  (the design originally specified a 2-lane mainline + zipper merge in
  Milestone 1.1, which worked but was superseded by a 1-lane + accel
  lane geometry in the network refresh; both layouts produced 0
  teleports at this demand)
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

## Phase 1 design decisions (as built)

The implemented scenario constants are pinned in
`configs/sumo/phase1_1.yaml` and the demand override in
`configs/experiments/dataset_constant_inflow.yaml`; those files are the
canonical source of truth and the bullets below are a descriptive
summary.

Physical scenario:
- Highway length: 2000 m
- Lanes: 1 (with 100 m acceleration lane downstream of the merge; this
  replaces the original Milestone 1.1 "2-lane mainline + zipper merge"
  layout)
- On-ramp position: 1300 m from upstream boundary
- On-ramp length: 200 m
- Speed limit: 120 km/h (33.33 m/s)
- Simulation duration: 3600 s (1 hour)
- Control step interval: 30 s → T_ctrl = 120 steps
- Detector spacing: 100 m starting at 100 m → N_x = 19 detectors along
  the mainline at positions [100, 200, …, 1900] m

Mainline demand (built):
- Single constant 1500 vph throughout every episode and across every
  training run. Ramp demand cap fixed at 800 vph.
- This is a simplification of the original Phase 1 plan, which called
  for a "small controlled family" of demand profiles (low ≈ 1000 vph,
  medium 1500, high 2000, mild peak ramping 1200 → 2200 → 1200) sampled
  per episode. The family is **deferred to Milestone 2c** and requires
  an M2 dataset rerun across demand levels plus a 240-dim DeepONet
  branch input (concatenating `mainline_demand(t)` and
  `ramp_control(t)`).

Tooling:
- DeepONet: pure PyTorch (no deepxde)
- PPO: Stable-Baselines3
- Simulation: SUMO with TraCI

Non-goals (Phase 1):
- Multi-ramp or multi-agent control
- Stochastic demand beyond the controlled family
- Incidents or disturbances
- Physics-informed loss in DeepONet
- Real-world calibration

## SUMO setup

Before running SUMO-based simulations in a fresh shell, make sure `sumo`,
`netconvert`, and `duarouter` are on `PATH` and the SUMO Python bindings
are importable.

**macOS (framework install):**
```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO"
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export PATH="$SUMO_HOME/bin:$PATH"
```

**Windows (Eclipse SUMO MSI install):** the installer registers
`SUMO_HOME` system-wide and adds `bin/` and `tools/` to `PATH`
automatically. Typical install path:
`C:\Program Files (x86)\Eclipse\Sumo\`. No shell exports needed.
To verify in PowerShell:
```powershell
echo $env:SUMO_HOME
Get-Command sumo, netconvert
```

If `sumo`, `netconvert`, or `duarouter` are "not found", the issue is
usually environment setup (PATH or SUMO_HOME) rather than a missing
installation.

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
│   ├── make_dataset.sh             # M2: runs sumo_env.dataset_generation
│   ├── train_surrogate.sh          # M3: runs surrogate.train
│   ├── train_ppo_surrogate.sh      # M5: runs rl.train_ppo with env.type=surrogate
│   ├── eval_in_sumo.sh             # M6: runs rl.evaluate on a saved PPO policy
│   ├── run_rollout.py              # M1: single-rollout CLI for manual SUMO checks
│   ├── inspect_rollout.py          # diagnostic: pretty-print one saved .npz rollout
│   ├── run_diagnostic_suite.py     # diagnostic: batch over a set of policies
│   ├── eval_constant_baselines.py  # M5b/M5c: roll out constant or learned policy in SurrogateEnv
│   ├── eval_sumo_baselines.py      # M6: SumoEnv counterpart of the above
│   └── run_m5b_sweep.py            # M5b/M5c: subprocess sweep driver over (beta, seed)
├── tests/
│   └── test_surrogate_env.py       # M4: pytest smoke tests for SurrogateEnv parity
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

**Implementation status:** This section describes the multi-demand
Phase 1 *design target*. The current Milestone 3 surrogate trains on a
single constant 1500 vph dataset and uses a **120-dim branch input
(ramp_control only)**, dropping the mainline_demand concatenation.
Upgrading to the 240-dim form below is **Milestone 3b**, blocked on
Milestone 2c building a multi-demand dataset.

Branch net input (design target — M3b):
- Concatenation of [ramp_control(t); mainline_demand(t)]
- Shape: (2 * T_ctrl,) = (240,)
- ramp_control values ∈ [0, 1] (metering rate)
- mainline_demand values normalized (min-max across the demand family)

Branch net input (currently implemented — M3):
- ramp_control(t) only
- Shape: (T_ctrl,) = (120,)
- ramp_control values ∈ [0, 1]
- Mainline demand is filtered to a single value (1500 vph) at dataset
  load time via `data.constant_mainline_demand_vph` in
  `configs/surrogate/baseline.yaml`

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
2. The branch input is constructed as:
   - **Design target (M3b, 240-dim):** `[u(0), ..., u(k), 0, ..., 0 ; d(0), ..., d(T-1)]` — ramp control first k+1 entries are actual actions, remainder zero-padded to T_ctrl; demand is the full known profile.
   - **Currently implemented (M3, 120-dim):** `[u(0), ..., u(k), 0, ..., 0]` only — the mainline-demand half is dropped because the MVP runs at a fixed single demand and the surrogate is trained on a demand-filtered subset of the dataset (`constant_mainline_demand_vph: 1500` in `configs/surrogate/baseline.yaml`).
3. The trunk queries all detector positions at time t_k: {(x_i, t_k) for i = 1..N_x}
4. The DeepONet returns density predictions at those points → this becomes the observation
5. The reward is computed from this density snapshot

This means the DeepONet is re-evaluated from scratch at every RL step (not autoregressive).

Known risk — distribution shift:
Training data contains fully-specified control signals. During RL rollout, partially-specified (zero-padded) signals are a distribution shift. The surrogate may produce unreliable density predictions for the zero-padded future portion, but we only query density at the current time t_k (not future times), which partially mitigates this.

Phase 1 training pipeline requirement (how it's actually implemented):
The zero-padded-prefix views the policy will see at RL time are
generated at **training time** by `surrogate.datasets.TrafficDataset`
via the `control_augmentation` block in
`configs/surrogate/baseline.yaml` (default: 16 padded prefix views
per train rollout, 0 for val/test). The raw M2 dataset on disk has
only full-control rollouts; the augmentation expands them into
~17× more training views without rerunning SUMO. For each padded
view, target query points are automatically restricted to
`t ≤ t_{prefix_len-1}` so the surrogate is never supervised on a
time index that depends on the zero-padded future portion of the
control signal. This is **not optional** — it is the mechanism that
lets the surrogate generalize to RL rollout conditions.

Additional mitigations:
1. Monitor surrogate prediction error during RL evaluation by comparing surrogate predictions against SUMO ground truth for the same control sequence
2. If transfer gap is large, consider autoregressive reformulation in Phase 2

## RL interface contract (Phase 1)

Observation design:
- Shape: (N_x + 3,) = (22,) with N_x = 19 detectors on the current `phase1_1.yaml` geometry
- Components:
  - density[0:N_x]: density at each detector at the current control step (z-score normalized, same normalization as surrogate training)
  - demand[N_x]: current mainline demand at this time step (min-max normalized to [0, 1])
  - time[N_x+1]: current normalized time index = k / T_ctrl ∈ [0, 1]
  - queue[N_x+2]: current on-ramp queue length, normalized by `queue_norm_scale` (default 100 vehicles). Unbounded; values can exceed 1.0 when queue is large.
- Justification: the agent needs the current traffic state (density field) to decide the metering rate, the current demand level to adapt across demand regimes, the time index because optimal metering strategy is time-dependent, and the queue length because the shaped reward penalizes queue buildup (without queue in the obs, a stateless MLP policy is partially observable on a term that directly affects reward). We do NOT include speed/flow (redundant given density), throughput (logged in info, not yet a reward term), or past actions (the surrogate handles temporal dependence internally via the full control history in the branch input).

Action:
- Shape: (1,)
- Semantics: ramp metering rate ∈ [0, 1]
  - 0.0 = ramp fully closed (no vehicles enter from on-ramp)
  - 1.0 = ramp fully open (all ramp demand enters)
- Continuous action space (Box)

Analytical queue model (shared by both envs):
- `queue[k+1] = max(0, queue[k] + (1 − u_k) · ramp_demand_vph · dt_ctrl_s / 3600)` with `queue[0] = 0`.
- Same formula in `SurrogateEnv` and `SumoEnv` for parity: the surrogate-vs-SUMO comparison in M6/M7 reflects only the density-dynamics gap, not a reward-signal gap.
- SUMO's measured queue length (`traci.edge.getLastStepVehicleNumber("ramp")`) is still recorded in the info dict for diagnostics but is not used as the reward signal.

Reward (Phase 1 shaped, Milestone 5c):
- Computed by a shared reward function `compute_reward(density, queue_length, weights)` in `src/rl/reward.py`, used identically by both surrogate and SUMO environments.
- Formula (nonlinear): `reward = -alpha · max(0, mean(density) - rho_freeflow) - beta · (queue_length / queue_norm)^2 - gamma · std(density)` with `density` in physical units (veh/km).
- Default weights: `alpha=1.0, beta=1.0, gamma=1.0, rho_freeflow=20.0, queue_norm=100.0`. All five are tunable per-experiment in the PPO config YAML.
- Term motivation:
  - `-alpha · max(0, mean(density) - rho_freeflow)`: ReLU penalty on density excess. Operation below free-flow density (mainline not yet congested) costs nothing; the term only activates when mean density crosses `rho_freeflow`, which on the current scenario corresponds roughly to the u=1.0 corner.
  - `-beta · (queue_length / queue_norm)^2`: quadratic penalty on queue. Cost grows fast as the queue builds, so long queues are heavily penalized while short queues are cheap. Replaces the linear queue term used in M5/M5b.
  - `-gamma · std(density)`: rewards spatially uniform density, discouraging local hotspots. Unchanged from M5.
- Empirical rationale: M5b's linear `-beta · queue_length` produced a structural corner trap at u=1.0 because the queue is identically 0 at that corner, making the term inactive regardless of beta. The ReLU-on-density + quadratic-on-queue shape breaks this by penalizing the u=1.0 corner specifically when its mean density crosses `rho_freeflow`, while still penalizing closer-to-closure policies non-linearly. See `_progress/milestone_5b_progress.md` §5b.5 for the linear-corner diagnostic and `_progress/milestone_5c_progress.md` for the M5c validation run.
- Future reward extensions (Phase 2+): quadratic ReLU on density excess (sharper congestion penalty), piecewise queue with a hard "unacceptable" threshold, throughput bonus, total travel time penalty. The current shape is still intentionally simple; weights and the two thresholds (`rho_freeflow`, `queue_norm`) can be retuned without code changes.

Episode structure:
- Episode length: T_ctrl = 120 steps (one full simulation horizon = 3600 s)
- At reset: sample a demand value from `env.demand_profiles` in the PPO config. The current MVP pins `demand_profiles: [1500.0]` (single-element list → degenerate sampling at 1500 vph every episode). The design target — sampling from a 4-element family of low / medium / high constant + mild peak profiles — is the Milestone 2c follow-up.
- No early termination in Phase 1

Environment parity:
- SurrogateEnv and SumoEnv must expose identical observation_space, action_space, and reward function
- PPO training code must be fully agnostic to which env it uses