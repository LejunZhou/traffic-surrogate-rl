# Formulation: SUMO → DeepONet surrogate → PPO ramp metering

This document states, in one place and one notation, what the code in
`src/` actually computes: how the SUMO data is generated, what the
DeepONet learns and how it is queried, the two Markov decision processes
(surrogate and SUMO) that PPO trains against, the PPO objective as
Stable-Baselines3 implements it, and how a trained policy is deployed
and evaluated. Numbers are the values the current configs resolve to
(`configs/sumo/phase1_1.yaml`, `configs/surrogate/baseline.yaml`,
`configs/rl/ppo_sumo_m7_run7_range.yaml`, `configs/rl/ppo_surrogate.yaml`).
Section 7 lists the points where the two paths diverge or the design is
open, for discussion.

---

## 1. Scenario and notation

| symbol | meaning | value |
|---|---|---|
| $L$ | mainline length | 2000 m |
| $x_i$ | detector positions, $i = 1..N_x$ | $100\,i$ m, $N_x = 19$ |
| $T$ | episode horizon | 3600 s |
| $\Delta t$ | control interval | 30 s |
| $K$ | control steps per episode | $T/\Delta t = 120$ |
| $t_k$ | start of control interval $k$ | $k\,\Delta t$, $k = 0..K-1$ |
| $d$ | mainline demand (constant per episode) | 1500–2000 vph (SUMO path), 2000 (surrogate path) |
| $r$ | ramp arrival rate (constant per episode) | 400/600/800 vph (SUMO path), 800 (surrogate path) |
| $D$ | ramp meter discharge capacity | 1600 vph |
| $u_k \in [0,1]$ | metering rate at step $k$ (green fraction of $D$) | action |
| $\rho_k \in \mathbb{R}^{N_x}$ | density at the detectors during interval $k$, veh/km | state |
| $Q_k \ge 0$ | virtual ramp queue (vehicles) at the end of interval $k$ | state |
| $q^{out}_k$ | mainline outflow during interval $k$, veh/h | reward input (SUMO only) |
| $\mu, \sigma$ | density normalisation (train-set mean/std) | 18.73, 5.971 veh/km |

One episode is one hour of traffic on an initially empty road. The
control $u_k$ is held constant over $[t_k, t_k + \Delta t)$.

**Ramp meter (both paths, since M7 §7.10).** Arrivals accumulate in a
virtual queue; the meter releases at most a fraction $u_k$ of $D$:

$$
a = \frac{r\,\Delta t}{3600},\qquad c_k = \frac{u_k\, D\, \Delta t}{3600},\qquad
\text{released}_k = \min(Q_{k-1} + a,\; c_k),\qquad
Q_k = Q_{k-1} + a - \text{released}_k .
$$

With $D = 1600$ and $r = 800$: $u = 0.5$ passes the full arrival rate,
$u = 1$ drains a backlog at +800 vph, and $Q$ grows for $u < r/D$.

---

## 2. Data generation in SUMO (`sumo_env/run_simulation.py`, `dataset_generation.py`)

**Inputs per rollout.** A demand cell $(d, r)$ cycled round-robin, a
seed, and a ramp control signal $\bar u = (u_0, \dots, u_{K-1})$ drawn
from one of four families with $u \sim U[0,1]$ levels: constant,
piecewise-constant (2–6 segments), smooth (4–8 linear knots), ramp-step.

**Simulation.** SUMO advances in 1 s steps. During interval $k$ only
$u_k$ is read: the metered queue (`MeteredRampQueue`) releases
$\min(u_k D, \text{queue})$ vehicles per hour onto the ramp edge; mainline
vehicles arrive at $d$ vph (`departSpeed="desired"`,
`--extrapolate-departpos`, wait-forever insertion). Formally, with
$s(t)$ the full SUMO state,

$$
s(t+1) = F\big(s(t),\, u_{\lfloor t/\Delta t\rfloor},\, \xi\big)
\;\;\Rightarrow\;\;
s(t) \text{ depends only on } u_0,\dots,u_{\lfloor t/\Delta t\rfloor}.
$$

**Labels.** Per detector and interval, from E1 induction loops
aggregated over the 30 sub-steps:

$$
q_{i,k} = \frac{\sum_{\text{sub}} n_{i}}{\Delta t}\cdot 3600,\qquad
v_{i,k} = \frac{\sum n_i\, \bar v_i}{\sum n_i},\qquad
\rho_{i,k} =
\begin{cases}
q_{i,k}/v_{i,k} & v_{i,k} > 5\ \text{km/h}\\
\text{occ}_{i,k}\cdot 1000/\ell_{veh} & \text{otherwise}
\end{cases}
$$

(`speed` and `flow` are stored for diagnostics; $\rho$ is the only
supervised target. Known caveat: $n_i$ over-counts by $\approx 1 + \ell/(v\,\Delta t_{sub}) \approx 1.2$ in free flow.)

**Stored per sample** (`sim_XXXX.npz`): `density` $(N_x, K)$, `speed`,
`flow`, `x_grid`, `t_grid`, `mainline_demand` $(K,)$, `ramp_control`
$(K,)$ = released inflow $/D$ (the branch input under `metered_queue`),
`ramp_control_cmd` = $\bar u$, `ramp_inflow_vph`, `ramp_queue`, scalars
$d, r$, seed. Split 70/15/15 by rollout; $\mu, \sigma$ computed on the
train split only.

**Currently available surrogate dataset.** 120 rollouts at
$d = 2000$, $r = 800$, open-loop ramp (`ramp_control` = $\bar u$, inflow
$= u \cdot 800$), generated on the pre-M7 Windows SUMO. The demand-range
config (360 rollouts over 6 × 3 cells, metered queue) exists but has not
been generated/trained on.

---

## 3. DeepONet surrogate (`surrogate/deeponet.py`, `datasets.py`, `train.py`)

### 3.1 Operator being learned

The surrogate approximates the causal solution operator of the traffic
system for a fixed initial condition (empty road) and fixed $(d, r)$:

$$
\mathcal{G}:\ \bar u \mapsto \rho(x, t), \qquad
\hat\rho(x,t) = \mathcal{G}_\theta(\bar u)(x,t)
= \sum_{p=1}^{P} B_p(\bar u)\; \mathcal{T}_p\!\left(\tfrac{x}{L}, \tfrac{t}{T}\right) + b_0 .
$$

- Branch $B: \mathbb{R}^{K} \to \mathbb{R}^{P}$, MLP $120 \to 512 \to 512 \to 512$, GELU.
- Trunk $\mathcal{T}: [0,1]^2 \to \mathbb{R}^{P}$, MLP $2 \to 512 \to 512 \to 512$, GELU.
- $P = 512$, scalar bias $b_0$. Output is the **normalised** density
  $z = (\rho - \mu)/\sigma$.

Mainline demand and ramp arrival rate are **not** inputs (single-cell
checkpoint). The design target (M3b) is a branch input
$[\bar u;\ d\text{-profile}]$ or $[\bar u;\ d_{norm};\ r_{norm}]$.

### 3.2 Training views and the prefix augmentation

Each training rollout $n$ with control $\bar u^{(n)}$ and labels
$z^{(n)}_{i,k}$ yields:

- **1 full view**: input $\bar u^{(n)}$, targets at all $(i, k)$.
- **16 padded views** with prefix lengths $m$ drawn without replacement
  from $\{1, \dots, K-1\}$:

$$
\bar u^{(n,m)} = (u_0, \dots, u_{m-1}, 0, \dots, 0),\qquad
\text{targets only at } k \le m-1 .
$$

The padded view is exactly the input the RL environment produces at
step $k = m-1$. Because $\rho_{i,k}$ depends only on $u_0..u_k$ (§2), the
labels of a padded view coincide with those of a real rollout whose
ramp closes after $t_{m-1}$; the augmentation therefore adds no
inconsistent supervision and the padded input lies on the physical
input manifold. Validation and test use full views only (plus 4 padded
views per rollout in `surrogate.eval`).

### 3.3 Objective and optimisation

Per view, 512 query points $(i,k)$ are resampled uniformly from the
allowed grid every epoch. Loss:

$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{\text{views}\in\mathcal{B}}
\frac{1}{512}\sum_{(i,k)}\Big(\mathcal{G}_\theta(\bar u)(x_i, t_k) - z_{i,k}\Big)^2 .
$$

Adam, lr $10^{-3}$, weight decay $10^{-6}$, batch 16 views, gradient
norm clip 1.0, 1000 epochs, validation MSE on the full grid every 10
epochs, `best.pt` = lowest validation MSE. Checkpoint stores weights,
config, and $(\mu, \sigma)$. Reported test relative $L_2$ = 0.074 (M3).

### 3.4 Inference inside the RL environment

At RL step $k$ the environment holds the history
$h_k = (u_0, \dots, u_k, 0, \dots, 0)$ and evaluates

$$
\hat z_{i,k} = \mathcal{G}_\theta(h_k)\!\left(\tfrac{x_i}{L}, \tfrac{t_k}{T}\right),\quad i = 1..N_x,
\qquad
\hat\rho_k = \max\!\big(\sigma \hat z_k + \mu,\ 0\big).
$$

One forward pass per step, 19 trunk points, no queries at $t > t_k$, no
state fed back. The model is re-evaluated from scratch every step, so
prediction error does not compound across steps. Conversely, the
observed state is never used as an input and the model cannot start
from a non-empty road.

---

## 4. The two environments as MDPs (`rl/surrogate_env.py`, `rl/sumo_env_wrapper.py`)

### 4.1 Observation (identical layout in both)

$$
o_k = \Big[\ \tfrac{\rho_{k-1} - \mu}{\sigma}\ \big|\ \tfrac{d - d_{min}}{d_{max} - d_{min}}\ \big|\ \tfrac{r - r_{min}}{r_{max} - r_{min}}\ \big|\ \tfrac{k}{K}\ \big|\ \tfrac{Q_{k-1}}{Q_s}\ \Big] \in \mathbb{R}^{N_x + 4}
$$

with $\rho_{-1} = 0$, $Q_{-1} = 0$, $Q_s = 100$ (`queue_norm_scale`).
The ramp-rate feature is present when `observe_ramp_demand: true`
(23 features; 22 without). The Box is unbounded; a jam gives z-scores
of 20–30.

### 4.2 Action

$u_k \in [0,1]$, `Box(1)`. With `symmetric_action: true` PPO acts on
$a_k \in [-1,1]$ and gymnasium maps $u_k = (a_k + 1)/2$; either way the
env clips to $[0,1]$.

### 4.3 Transition

| | Surrogate env | SUMO env |
|---|---|---|
| density $\rho_k$ | $\mathcal{G}_\theta(h_k)(x_i, t_k)$, §3.4 | E1 aggregation over the 30 SUMO steps, §2 |
| queue $Q_k$ | closed-form recursion of §1 once per step | same recursion at 1 s resolution with integer accumulators; decremented only when SUMO reports the vehicle departed; reward uses the interval mean, observation the end value |
| outflow $q^{out}_k$ | not available | `getArrivedNumber` summed over the interval $\times 3600/\Delta t$ |
| stochasticity | none (deterministic model, fixed demand cell) | seed per episode, `speed_dev` 0.03, demand cell sampled per episode |
| branch input written | $h_k[k] = u_k$ (`action`) or released$_k / c_k$ (`inflow_frac`) | n/a |

Episodes terminate at $k = K$; no truncation, no early termination.

### 4.4 Reward (`rl/reward.py`, shared)

$$
r_k = -\,\delta\,\frac{\max(0,\ q_{ref} - q^{out}_k)}{q_{ref}}
      \;-\;\beta\left(\frac{Q_k}{Q_n}\right)^{2}
      \;-\;\gamma\,\frac{\operatorname{std}_i(\rho_{i,k})}{\sigma_{ref}},
\qquad
r_k \leftarrow 0 \ \text{if}\ t_k < t_{warm}.
$$

| weight | SUMO (run 7) | surrogate (`ppo_surrogate.yaml`) |
|---|---|---|
| $\delta$ / $q_{ref}$ | 3.572 / 2476 vph | **0** / – (no outflow prediction) |
| $\beta$ / $Q_n$ | 1.0 / 400 | 1.0 / **200** |
| $\gamma$ / $\sigma_{ref}$ | 0.063 / 6.0 | 1.0 / **67.72** |
| $t_{warm}$ | 90 s | 0 |

Each term is a non-positive, $O(1)$-per-step penalty; the SUMO weights
were balanced over the demand grid so that no term's episode-sum range
dominates. The reward is instantaneous by design; all future
consequences enter through the critic (§5).

---

## 5. PPO training (`rl/train_ppo.py`, Stable-Baselines3)

### 5.1 Policy and critic

Diagonal Gaussian policy with a state-independent log-std:

$$
\pi_\theta(a \mid o) = \mathcal{N}\big(\mu_\theta(o),\ \operatorname{diag}(e^{2\,\log\sigma})\big),\qquad
V_\phi(o) \in \mathbb{R},
$$

$\mu_\theta$ and $V_\phi$ are separate MLPs $[\,512, 512, 512\,]$ with
tanh. $\log\sigma$ is a free parameter, initialised to $-2$ on the SUMO
path (`log_std_init`) and $0$ on the surrogate path. `action_init_u`
sets the output-layer bias so that $\mu_\theta(o) = 2u_0 - 1$ at
initialisation ($u_0 = 0.3$).

### 5.2 Rollout and advantage estimation

Collect $n_{steps} = 480$ transitions (4 episodes) with the current
$\pi_{\theta_{old}}$, then

$$
\delta_k = r_k + \gamma V_\phi(o_{k+1})\,[k+1 < K] - V_\phi(o_k),\qquad
\hat A_k = \sum_{j \ge 0} (\gamma\lambda)^j \delta_{k+j},\qquad
\hat R_k = \hat A_k + V_\phi(o_k),
$$

$\gamma = 0.99$, $\lambda = 0.95$, no bootstrap past the terminal step.
This is where "$r + V$" lives: $\hat A_k$ measures whether $u_k$ made
$r_k + \gamma V(o_{k+1})$ exceed $V(o_k)$.

### 5.3 Update

For 5 epochs over minibatches of 120, with $\hat A$ normalised to zero
mean / unit variance per minibatch and $\varrho_k = \pi_\theta(a_k|o_k)/\pi_{\theta_{old}}(a_k|o_k)$:

$$
\mathcal{L}(\theta,\phi) =
-\,\mathbb{E}\Big[\min\big(\varrho_k \hat A_k,\ \operatorname{clip}(\varrho_k, 1-\epsilon, 1+\epsilon)\hat A_k\big)\Big]
+ c_v\,\mathbb{E}\big[(V_\phi(o_k) - \hat R_k)^2\big]
- c_e\,\mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot|o_k))\big].
$$

| hyper-parameter | SUMO (run 7) | surrogate |
|---|---|---|
| $\epsilon$ (clip) | 0.2 | 0.2 |
| $c_v$, $c_e$ | 0.5, 0 | 0.5, 0.05 → 1e-4 (exponential schedule) |
| learning rate | 1e-4 | 1e-4 |
| `target_kl` (early stop of the epoch loop when approx-KL exceeds it) | 0.02 | none |
| `max_grad_norm` | 0.5 | 0.5 |
| $n_{steps}$ / minibatch / epochs | 480 / 120 / 5 | 120 / 120 / 5 |
| total timesteps | 80 000 (≈ 667 episodes) | 1 000 000 (≈ 8 333 episodes) |
| `symmetric_action`, `action_init_u` | true, 0.3 | not set (PPO starts at $u \approx 0$) |

### 5.4 Checkpoint selection (SUMO path)

Every 2400 steps a deterministic evaluation ($a = \mu_\theta(o)$) runs
on a separate SUMO env that cycles all 18 demand cells with fixed
seeds; `best_model.zip` tracks the best mean. Every evaluated
checkpoint is also saved, and `scripts/select_checkpoint_multiseed.py`
re-scores the top-5 on 18 cells × 3 seeds and picks the best grid mean
subject to no episode below −150 (`best_model_multiseed.zip`).

---

## 6. Deployment and evaluation

**Policy at inference.** Deterministic:
$u_k = \operatorname{clip}\!\big(\tfrac{\mu_\theta(o_k) + 1}{2},\ 0,\ 1\big)$
under the symmetric mapping. The DeepONet is not used at deployment;
it is a training-time environment only. A surrogate-trained policy is
run in `SumoEnv` exactly like a SUMO-trained one; evaluation scripts
read the action-space bounds from the saved model to undo the mapping.

**Evaluation protocol** (`scripts/eval_policy_grid_sumo.py`,
`analyze_policy_grid.py`): 18 cells × 3 seeds, `speed_dev` 0.03, full
three-term reward. Reported: grid-mean return, worst episode, 10th
percentile, episodes below −150, breakdown/recovery counts,
throughput and final queue, $u(d, r)$ sensitivities. Baselines:
constant $u$, ALINEA, PI-ALINEA.

**Transfer metric (the research question).** For a policy trained in
env $E \in \{\text{surrogate}, \text{SUMO}\}$, compare
$J_{SUMO}(\pi_E)$ under the same reward, plus wall-clock and PPO
iterations to reach it. The surrogate-vs-SUMO density gap can be
measured directly by replaying a SUMO episode's $\bar u$ through
$\mathcal{G}_\theta$ (M6 §6.6 did this for constant policies: ≤ 4 %
on episode return).

---

## 7. Points for discussion (where the two paths currently diverge)

1. **Reward mismatch.** The surrogate path runs $\delta = 0$,
   $\sigma_{ref} = 67.7$: effectively a queue-only reward that is flat
   for all $u \ge 0.5$. A policy trained there optimises a different
   objective than the SUMO policy. Fix requires an outflow prediction
   (second output head, or flow at $x_{19}$ from a joint $(\rho, q)$
   operator) and matching weights.
2. **Action semantics of the checkpoint.** The M3 model saw
   `ramp_control` = inflow/800; the current env's $u$ is inflow/1600.
   Feeding $u$ directly mis-indexes by 2×. Needs retraining on the
   metered-queue dataset with `inflow_frac`.
3. **Scenario mismatch.** The M3 dataset came from a SUMO that
   inserted ≈ 1470 vph; the current scenario inserts 2000 vph with
   `speed_dev` 0.03 and seed-dependent breakdowns.
4. **No demand conditioning.** Branch input is $\bar u$ only; the
   demand-range setting needs $[\bar u; d_{norm}; r_{norm}]$ (122-dim)
   and relaxing the `branch_input_dim == K` check.
5. **Zero-padding convention.** Correct for the target by causality
   (§2) and consistent under augmentation (§3.2), but the branch MLP
   must *learn* to ignore slots $> k$. Unmeasured: per-prefix-length
   error and sensitivity of $\hat\rho_k$ to the padded slots. Structural
   alternative: causal branch (1-D causal conv / GRU read at index $k$).
6. **Deterministic operator vs stochastic simulator.** $\mathcal{G}_\theta$
   regresses to the conditional mean and smooths the capacity cliff at
   which the SUMO policy has to operate. Options: multi-seed labels,
   noise injection in the surrogate env, a breakdown-onset metric.
7. **History-to-state vs state-to-state surrogate.** The current design
   has no error compounding and a sufficient input in the deterministic
   case, but cannot use observations or arbitrary initial states. A
   one-step model $s_{k+1} = f(s_k, u_k, d, r)$ is the standard
   alternative; a hybrid (branch = current $\rho_k$ + recent actions,
   trunk = next window) keeps the field output needed for PI-DeepONet.
   Any change here is a proposal-level decision.
8. **Surrogate PPO config lags the SUMO one**: no `symmetric_action`,
   no `action_init_u`, no `target_kl`, no deterministic eval, $Q_n = 200$
   vs 400, `n_steps` 120 vs 480.
9. **Finite horizon and the time feature.** No terminal cost on $Q_K$
   or on a jam at $t = T$; the policy can legitimately open the ramp late
   in the hour. Check run 7's last 10 minutes; add a terminal penalty or
   drop $k/K$ if a stationary controller is wanted.
10. **Observation scale.** Unbounded z-scores (≈ 30 in a jam) into tanh
    layers; consider clipping or a log transform before retraining.
