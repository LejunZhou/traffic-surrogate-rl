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
- 2-lane mainline highway + single-lane on-ramp
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
- Lanes: 2
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
- Real-world calibration

## SUMO setup on this Mac

SUMO is installed at `/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO`.

Before running SUMO-based simulations in a fresh shell, set these environment variables:
```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO"
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export PATH="$SUMO_HOME/bin:$PATH"
```

If `sumo`, `netconvert`, or `duarouter` are "not found", the issue is usually environment setup, not a missing installation.

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

Reward (Phase 1 shaped):
- Computed by a shared reward function `compute_reward(density, queue_length, weights)` in `src/rl/reward.py`, used identically by both surrogate and SUMO environments.
- Formula: `reward = -alpha · mean(density) - beta · queue_length - gamma · std(density)` with `density` in physical units (veh/km).
- Default weights: `alpha=1.0, beta=0.1, gamma=1.0`. Tunable per-experiment in the PPO config YAML.
- Term motivation:
  - `-alpha · mean(density)`: penalizes mainline congestion (the original Phase-1 baseline).
  - `-beta · queue_length`: penalizes the on-ramp queue, so closing the ramp entirely is no longer free.
  - `-gamma · std(density)`: rewards spatially uniform density, discouraging local hotspots.
- Empirical rationale: the original `alpha=1, beta=0, gamma=0` baseline converged to the degenerate `u≡0` policy because the reward had a corner solution at "close the ramp completely" (Milestone 5 finding, archived on `mvp-v1-old-scenario`). The queue term creates the counterbalance that makes the metering problem non-trivial.
- Future reward extensions (Phase 2+): throughput bonus, total travel time penalty, fairness across vehicles. The current shaped reward is still intentionally simple; weights can be retuned without code changes.

Episode structure:
- Episode length: T_ctrl = 120 steps (one full simulation horizon = 3600 s)
- At reset: sample a demand profile from the controlled family
- No early termination in Phase 1

Environment parity:
- SurrogateEnv and SumoEnv must expose identical observation_space, action_space, and reward function
- PPO training code must be fully agnostic to which env it uses