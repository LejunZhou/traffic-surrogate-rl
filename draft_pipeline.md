# Draft pipeline: surrogate-in-the-loop ramp metering under time-varying demand

Status: proposal for discussion (2026-09-09). Nothing in this document is
implemented yet. Where it disagrees with the current code or configs it says
so and gives the reason. Section 13 lists the decisions that need the user.

---

## 0. What this document is

The project set out to show that a DeepONet surrogate of a SUMO on-ramp merge
makes PPO ramp-metering training cheaper without losing the policy quality one
gets from training in SUMO directly. Seven milestones later the pieces exist
(SUMO scenario, dataset generator, DeepONet, SurrogateEnv, SumoEnv, PPO,
ALINEA baselines, a grid evaluation protocol), but they were built one at a
time and the surrogate path no longer matches the SUMO path on scenario,
action semantics, reward, or PPO settings. The headline "surrogate beats SUMO
training" result (M6) has also been overtaken: after the action-space and
exploration fixes of M7, PPO trained directly in SUMO produces a good policy
(run 7) in about three hours.

So the question is no longer "can the surrogate replace SUMO" but "where does
an operator-learning surrogate genuinely pay for itself, and how do we set the
study up so that the answer is measurable and publishable". This document
proposes that setup for the case the user wants next: mainline demand and ramp
arrival rate that vary *within* the hour.

The proposal in one paragraph: treat the DeepONet as a learned **plant model**
(boundary inflows in, density and outflow fields out), train it on a small,
deliberately mixed set of closed-loop SUMO rollouts, wrap it as a batched,
ensemble-randomised Gym environment with the *same* reward and observation as
SUMO, train PPO on it, and close the loop with a budgeted **data-aggregation**
cycle (a few SUMO rollouts of the current policy per round, appended to the
surrogate's training set). Every SUMO rollout is counted. The paper's central
figure is held-out SUMO performance versus cumulative SUMO episodes for
(a) direct PPO in SUMO, (b) surrogate PPO with aggregation, (c) surrogate PPO
plus short SUMO fine-tuning, against tuned ALINEA and a surrogate-based MPC.

---

## 1. Audit: what exists, what it proves, what it cannot do

### 1.1 Pipeline as built (facts from the repo)

| Stage | Implementation | Key numbers |
|---|---|---|
| Scenario | `configs/sumo/phase1_1.yaml`: 1-lane mainline 2000 m, on-ramp at 1300 m, 100 m acceleration lane, 19 E1 loops at 100 m spacing, 30 s control step, K = 120 steps/h, empty road at t = 0 | merge capacity ~2480 vph total (ramp ≤ 500 vph), ~2300 with 800 vph ramp; gridlock discharge ~1690 vph; breakdown effectively irreversible at 2000 vph mainline |
| Ramp meter | virtual queue, arrivals r, discharge 1600 vph, release = min(Q + a, u·1600) (`MeteredRampQueue`, same recursion in both envs) | u = r/1600 passes demand; u = 1 flushes at +800 vph |
| Demand | constant per episode: mainline 1500–2000, ramp 400/600/800 (18 cells), `speed_dev` 0.03 for seed variability | free-flow returns are seed-invariant; only edge cells vary |
| Dataset generator | `sumo_env.dataset_generation` (open-loop control families) plus an untracked family-schema generator used for the 6000-rollout shockwave set on another machine (`run_parallel_generation.py` imports `build_generation_plan`, which is not in this tree) | M2: 120 rollouts; shockwave: 6000 rollouts at 2000 vph |
| DeepONet | branch MLP on the full 120-step control vector (zero-padded prefix at RL time), trunk MLP on (x, t), p = 512, density only, z-scored; prefix augmentation (16 views/rollout) | M3 rel-L2 0.074 (free-flow data from the old SUMO); shockwave retrain rel-L2 0.235 global, 0.21 in the shockwave band, 0.29 on bang-bang controls |
| SurrogateEnv | re-evaluates the DeepONet from scratch each step with the padded action history; analytic queue; reward with delta = 0 (no outflow) | ~0.2 s per episode |
| SumoEnv | TraCI, exact arrival-count outflow, pending/discard bookkeeping, labelled connections | 13–30 s per episode |
| PPO | SB3, 3×512 tanh MLPs, symmetric action box, `action_init_u`, `log_std_init` −2, `target_kl` 0.02, EvalCallback cycling all 18 cells, post-hoc multi-seed checkpoint selection | run 7: 80k steps ≈ 667 episodes ≈ 2.7 h |
| Baselines | ALINEA / PI-ALINEA tuned at the merge nose, constant u | ALINEA −62.7 vs run 7 −60.7 (grid mean), identical worst case |

### 1.2 Results that stand

1. The scenario is physically sound and reproducible: exact insertion at every
   demand level, jams recover after closure, SUMO version pinned (1.27.1).
2. The three-term reward is well posed: interior optimum, balanced over the
   demand grid, and the learned policy does genuine feedback control (throttles
   on mainline demand, re-opens with queue pressure, cuts u on jam onset).
3. Direct SUMO PPO works once the action box is symmetric and exploration is
   narrow; multi-seed checkpoint selection is required at the capacity edge.
4. Tuned ALINEA is within ~2 return units of the learned policy. The learned
   policy's edge is *pre-emptive* metering on high-ramp cells (11 vs 27
   transient jams), which it gets from observing queue and demand.
5. The DeepONet reproduces SUMO episode returns of constant policies to within
   4 % in free flow (M6 §6.6). It has not been validated in the breakdown
   regime with closed-loop controllers, and its shockwave-regime accuracy is
   mediocre.

### 1.3 Why the current surrogate path cannot be used for the M7 benchmark

These are the concrete mismatches. Each one is fixed by a design decision in
Section 3.

| # | Gap | Consequence |
|---|---|---|
| G1 | Branch input is the *action* u (fraction of 800 vph in the M3 data; the env's u is a fraction of 1600) | the surrogate is driven with a mis-indexed control; `inflow_frac` mode exists but no checkpoint was trained on metered-queue data |
| G2 | No outflow prediction, so delta = 0 on the surrogate path | the surrogate policy optimises a queue + std reward that is flat for all u ≥ r/D; it is a different objective from the SUMO policy's |
| G3 | No demand conditioning (branch = u only) | one checkpoint per demand cell; time-varying demand is impossible |
| G4 | Single deterministic network trained by MSE | near the capacity edge it regresses to the mean of jam / no-jam outcomes; PPO trained on it parks on the edge and gridlocks in SUMO (run 4 pattern) |
| G5 | Training controls are random open-loop families | the distribution the *policy* induces (edge-riding, recovery closures, store-and-flush) is under-represented; bang-bang, the family that dominates the shockwave set, is the one the model fits worst |
| G6 | Dataset cost is not accounted for | the 6000-rollout shockwave set costs 9× the SUMO budget of run 7; a sample-efficiency claim cannot be made on top of it |
| G7 | Surrogate PPO config lags the SUMO one (no symmetric action, no `action_init_u`, no `target_kl`, queue_norm 200 vs 400, sigma_ref 67.7 vs 6, n_steps 120 vs 480) | the surrogate-vs-SUMO comparison is confounded by everything except the environment |
| G8 | Density estimators differ in the jam regime (q/v over-count ~1.2× in free flow; occupancy fallback saturates at 200 veh/km/lane and sums the two accel-lane loops) | the surrogate target and the SUMO observation are not the same quantity in exactly the regime that matters |
| G9 | The fixed q_ref = 2476 outflow term becomes a constant offset whenever offered demand is below 2476 | under time-varying demand the term penalises low-demand minutes the controller cannot influence: pure noise in the critic target |

### 1.4 Honest status of the surrogate claim

"Faster" is true but not sufficient (0.2 s vs 13–30 s per episode, ~100×).
"Enables learning that SUMO cannot" (M6) was an artefact of the action-space
initialisation and is no longer true. What remains defensible, and what
Section 2 turns into claims: the surrogate is worth it when (i) the policy
search needs many more episodes than a constant-demand grid does, (ii) the
same plant model is reused across many RL runs (seeds, hyper-parameters,
reward variants, observation variants), and (iii) it gives things SUMO cannot
give at all: batched rollouts, differentiability with respect to the inflow
history (MPC, sensitivity), and a continuous space-time field.

---

## 2. Research question, claims, and the budget ledger

**Research question.** For ramp metering under time-varying mainline demand
and ramp arrivals, does an operator-learning plant model (DeepONet) trained
on a small budget of SUMO rollouts let PPO reach the policy quality of direct
SUMO training with fewer SUMO episodes, and does it transfer to held-out and
out-of-distribution demand profiles?

**Claims the pipeline is designed to test** (each maps to an experiment in
Section 11):

- C1 (sample efficiency). At matched SUMO-episode budget, surrogate PPO with
  aggregation reaches a higher held-out SUMO return than direct SUMO PPO, and
  surrogate pre-training plus a short SUMO fine-tune reaches direct-SUMO
  quality with ≤ 1/3 of its SUMO episodes.
- C2 (transfer). Policies trained on the ensemble surrogate keep their
  breakdown rate when moved to SUMO; policies trained on a single
  deterministic surrogate do not.
- C3 (generalisation). The surrogate-trained policy beats tuned ALINEA on
  held-out profiles and degrades gracefully on an out-of-distribution profile
  family; the anticipative observation variant (10-min demand look-ahead)
  widens the margin.
- C4 (what the operator model adds). DeepONet's history-to-field formulation
  gives lower long-horizon error and better return prediction than a
  one-step autoregressive dynamics model of the same size, and its
  differentiability yields a strong MPC baseline for free.
- C5 (amortisation). One plant model supports the PPO seed sweep, the
  reward-variant ablation, and the observation ablation at a small fraction
  of their SUMO cost.

**Budget ledger.** The unit is one SUMO episode-equivalent (EE): one 1-h
rollout of the scenario, regardless of who drives it. Counted: dataset
rollouts, aggregation rollouts, fine-tuning episodes, direct-PPO training
episodes, controller-tuning episodes (for ALINEA). Not counted: the final
test-set evaluation (identical for every method). Wall-clock is reported
separately and includes surrogate training and inference. Every figure that
compares methods has EE on its x-axis or in its caption.

---

## 3. Design decisions (and where they disagree with the current setup)

Each decision states what changes, why, and what it costs.

**D1. The surrogate learns the plant, not the plant-plus-meter.**
Branch input becomes the two *physical boundary inflows*: mainline demand
d(t) and released ramp inflow q_r(t), both as K-step sequences. The action u
never enters the surrogate; the environment converts u into q_r through the
analytic queue exactly as SumoEnv does. Why: the mainline dynamics depend on
what enters the road, not on the meter's green fraction; this removes the
u-vs-inflow_frac ambiguity (G1) permanently, makes the surrogate independent
of the discharge capacity and of any future change to the meter, and is the
same quantity the metered-queue generator already stores (`ramp_inflow_vph`).
Cost: none beyond a new dataset, which is needed anyway.

**D2. The surrogate predicts outflow as well as density.**
Second output channel q(x, t); the reward's outflow is q at the exit, trained
against the exact arrival count (new label `outflow_vph[k]` written by the
generator, same definition as SumoEnv). Why: reward parity (G2). Without it
the surrogate policy optimises a different objective and the comparison is
meaningless. Cost: one more label and a two-channel DeepONet head.

**D3. Demand-conditioned, time-varying inputs.**
Branch input dimension 2K (d and q_r) instead of K; both channels prefix-
padded at RL time, because ρ(x, t_k) depends only on inflows up to t_k. Why:
(G3) and the user's target setting. This is exactly the M3b design target in
the proposal (240-dim branch), with q_r in place of u.

**D4. Causal branch encoder as the default, zero-padded MLP as the ablation.**
Replace the branch MLP over the padded vector with a causal 1-D convolutional
(or GRU) encoder whose read-out at index k depends only on inputs ≤ k; the
trunk is unchanged. Why: the padded-MLP must *learn* to ignore future slots
and its per-prefix error has never been measured (formulation.md §7.5); a
causal encoder makes prefix padding unnecessary and puts every RL-time input
on the training manifold by construction. It is still a DeepONet (the branch
architecture is free in the original formulation). Cost: one module, one
config switch; the prefix-augmented MLP stays as the ablation.

**D5. Ensemble of surrogates trained on stochastic rollouts; one member per
episode during RL.** M = 5 DeepONets (different seeds, bootstrap resamples),
each trained by MSE on rollouts generated with `speed_dev` 0.03 and random
SUMO seeds. The RL env samples a member at reset. Why (G4): a single MSE
network smooths the capacity cliff into a fictitious intermediate state; an
ensemble presents the policy with plausible plants that disagree exactly
there, so knife-edge operation is penalised in expectation (the MBPO/PETS
argument). Ensemble disagreement is also the diagnostic for "where the
surrogate does not know" that drives aggregation (D7). Cost: 5× surrogate
training (parallelisable) and 5× inference, which is negligible.

**D6. Round-0 dataset from a behaviour-policy mixture, not from random
open-loop controls.** Composition in Section 6.1: constant-u anchors,
ALINEA with randomised gains and set-points, structured store-and-flush
schedules, the run-7 policy with action noise, and a minority of random
piecewise / smooth signals. Why (G5): the surrogate must be accurate where
the policy will go, and closed-loop controllers produce the edge-riding,
jam-onset and recovery trajectories that random signals produce only by
accident. The 35 % bang-bang share of the shockwave set is the hardest family
for the model and the least relevant to control. Cost: the generator needs a
"controller" mode (roll a callable policy through `run_simulation`).

**D7. Budgeted data aggregation.** After each PPO phase on the surrogate, the
top checkpoints are rolled in SUMO on the validation profiles (this is the
transfer measurement we need anyway); those rollouts are appended to the
surrogate dataset and the ensemble is fine-tuned. Why (G5, G6): it is the
standard, principled fix for model-based RL distribution shift, and it makes
the SUMO cost of closing the sim-to-sim gap explicit and small. Cost: the
orchestration script.

**D8. One PPO configuration; the environment type is the only switch.**
Symmetric action box, `action_init_u`, `log_std_init` −2, `target_kl` 0.02,
lr 1e-4, n_steps = 4 episodes, the same reward weights, same queue_norm and
sigma_ref, same observation layout, same evaluation cycling over a fixed
validation set. Why (G7): otherwise the environment comparison is confounded.
Cost: config consolidation.

**D9. One density estimator in both paths: occupancy-based with the
minGap-corrected effective length, lane-averaged on the acceleration
segment.** ρ = occ · 1000 / (ℓ + minGap) with ℓ + minGap = 7 m gives a jam
density of ~143 veh/km/lane, bounded, no q/v blow-up, no over-count. The
q/v estimate and raw loop flow stay in the npz for diagnostics. Why (G8): the
surrogate target and the policy observation must be the same quantity in
every regime. Cost: a config key already sketched in the pilot config
(`detectors.density_method: occupancy`) plus the minGap correction and
lane-averaging; the run-7 policy's observation statistics change, so it must
be re-normalised before use as a behaviour policy.

**D10. Outflow term measured against offered demand, not a fixed capacity.**
q_ref,k = min(d_k + r_k + drain allowance, q_cap) with q_cap = 2476; the
"offered" mode already exists in the balancing script. Why (G9): under
time-varying demand the term should measure *unserved* demand each step; a
fixed reference penalises quiet minutes and adds variance to the critic
target. Re-balance delta / beta / gamma once on the profile family with the
constant-u sweep plus the store-and-flush schedules (Section 7.2). Cost: a
reward-config mode and one balance run.

**D11. Profiles end with a drain tail; a terminal queue cost is an option,
not the default.** Every profile has its mainline peak finished by minute 45
so a store-and-flush policy has time to release its queue inside the hour.
Why: a finite horizon with no terminal cost lets the policy dump the queue at
t = 60 min for free (formulation.md §7.9); the physical fix (a tail) keeps
the reward unchanged. A terminal cost β_T (Q_K / Q_n)² is kept as an ablation
flag.

**D12. Keep the history-to-field formulation; do not switch to an
autoregressive state model for the core pipeline.** Episodes always start
from an empty road, the inflow history is a sufficient input, there is no
error compounding, and the formulation is what makes MPC re-planning and the
PI-DeepONet extension natural. A one-step MLP/GRU dynamics model is built
only as the *baseline surrogate* the paper needs to justify DeepONet (C4).

**What is deliberately kept:** the scenario geometry and SUMO settings, the
30 s control step, K = 120, the virtual queue with discharge 1600, the
three-term reward form and its balancing procedure, the SB3 PPO stack, the
multi-seed checkpoint selection, and the ALINEA baselines and their tuning
protocol.

---

## 4. Scenario: time-varying mainline demand and ramp arrivals

### 4.1 Profile families

All profiles are piecewise-constant on 5-min blocks (12 blocks per hour,
10 control steps per block), which SUMO can realise exactly with one `<flow>`
element per block and which keeps the profile parameter space small.

Mainline demand d(t):

| Family | Form | Parameters (sampled per episode) |
|---|---|---|
| M-peak | d_base + A · bump((t − t_c) / w), raised-cosine bump | d_base ~ U[1200, 1700], A ~ U[300, 900], t_c ~ U[15, 35] min, half-width w ~ U[7.5, 15] min; clip d ≤ 2300; peak must end by minute 45 |
| M-step | d_base → d_high over a 10-min ramp at t_1, optionally back down at t_2 | d_base ~ U[1200, 1600], d_high ~ U[1800, 2300], t_1 ~ U[5, 25] min, t_2 ∈ {none, t_1 + U[10, 20] min} |
| M-const (10 % of draws) | constant | d ~ U[1500, 2000] (keeps the M7 grid inside the family) |

Ramp arrivals r(t):

| Family | Form | Parameters |
|---|---|---|
| R-surge | r_base + A_r on [t_r, t_r + L_r] | r_base ~ U[200, 500], A_r ~ U[0, 500], t_r ~ U[5, 40] min, L_r ~ U[10, 20] min; clip r ≤ 1000 |
| R-const (20 %) | constant | r ~ U[300, 800] |

The mainline peak is capped at 2300 vph, below the ~2480 vph merge capacity,
so that the ramp can always be served *if* the meter stores during the peak;
peak totals d + r span roughly 1500–3300 vph, so every regime from
pass-through to mandatory storage occurs. Surge timing is sampled
independently of the mainline peak, so both "ramp surge during the mainline
peak" (store) and "surge before it" (anticipate) occur.

Out-of-distribution family for evaluation only: M-double (two peaks), M-plateau
(2200 vph for 30 min), and R-surge starting before the mainline peak with
A_r = 600.

### 4.2 Sets and splits

| Set | Content | Use |
|---|---|---|
| Training family | the parametric families above, resampled every episode | surrogate data, PPO on surrogate, direct PPO in SUMO |
| Validation set V | 18 fixed profiles from the family (stratified over peak total and surge timing), fixed SUMO seeds 10000+ | EvalCallback cycling, checkpoint ranking, aggregation rollouts |
| Test set T | 30 fixed profiles from the family, seeds 100/101/102 (90 episodes) | all headline numbers; never used for selection or tuning |
| OOD set O | 12 profiles from the OOD families, 3 seeds | generalisation |

The profile sampler is seeded by (set, index) so every method sees identical
profiles; the sampler lives in `sumo_env/demand_profiles.py` and is shared by
the generator, SumoEnv and SurrogateEnv.

### 4.3 SUMO implementation

- `network_builder._write_routes`: write 12 `<flow>` elements (begin/end per
  block, `vehsPerHour` = d_block) with the existing `departSpeed="desired"`
  and `--extrapolate-departpos`. Verify exact insertion per block with the
  existing `check_demand_range_sumo.py` logic extended to profiles (E0).
- Routes are now per-profile, so they are written per episode into the env's
  network dir (cheap: a 30-line XML).
- Ramp arrivals: `MeteredRampQueue` gets a per-step `arrival_vph` instead of
  a constant; SumoEnv's inline accumulator likewise.
- The generator writes `mainline_demand (K,)` (already), `ramp_arrival (K,)`,
  `ramp_inflow_vph (K,)`, `ramp_queue (K,)`, `outflow_vph (K,)` (from
  `getArrivedNumber`, new), `pending_mainline (K,)` (diagnostic), the profile
  parameters, the behaviour-controller spec, seed and `speed_dev`.

### 4.4 Scenario characterisation (experiment E0, SUMO only)

Before any learning: (i) per-block insertion check on 20 random profiles;
(ii) capacity map: constant u ∈ {0, 0.1, …, 0.6} on 12 profiles, recording
breakdown onset and recovery; (iii) the store-and-flush schedule family
(close to u_low during the peak, flush at u = 1 after) swept over
(u_low, flush start) on the same profiles, to confirm that a dynamic policy
beats every constant one by a margin worth learning (expected: 20–40 return
units on peaked profiles, since the M7 grid showed the optimal constant moves
from u = 1 to u = 0.3 across the demand range). If (iii) fails, the profile
family is too mild and A / A_r must be raised.

---

## 5. Plant model: the DeepONet operator

### 5.1 Operator

For a fixed geometry and an empty initial road, the surrogate approximates
the causal solution operator

    G : (d(·), q_r(·)) ↦ (ρ(x, t), q(x, t)),   0 ≤ x ≤ L, 0 ≤ t ≤ T,

trained on samples of the stochastic plant (IDM with speed_dev 0.03). Each
ensemble member is one regression to the conditional mean; the ensemble
spread approximates predictive uncertainty. Inputs are given at the K-step
resolution (d and q_r per control interval), outputs are queried at the 19
detector positions per interval during RL and anywhere on the grid during
evaluation.

### 5.2 Inputs, outputs, normalisation

| Tensor | Shape | Normalisation |
|---|---|---|
| branch: d(t_0..t_{K−1}) | (K,) | / 2500 vph |
| branch: q_r(t_0..t_{K−1}) | (K,) | / 1600 vph (the `inflow_frac` convention) |
| trunk: (x, t) | (·, 2) | x / L, t / T |
| output: ρ(x, t) | (·,) | z-score with train-split mean/std |
| output: q(x, t) | (·,) | / 2500 vph; the exit value q(L, t) is the outflow label |

Prefix padding (for the MLP-branch ablation) zeroes both channels for
indices > k and restricts supervision to t ≤ t_k; the causal branch needs
no padding and is supervised on every (x, t_k) with the full sequence
in a single pass.

### 5.3 Architecture

- Branch A (default): causal encoder. 1-D convolution stack over the
  (2, K) input with left padding only (kernel 5, dilations 1/2/4/8, 128
  channels, GELU), read out at every k → (K, p). A GRU variant is a one-line
  alternative. Output per k is the branch vector used for queries at t_k.
- Branch B (ablation, current design): MLP on the 2K padded vector → p.
- Trunk: MLP 2 → 512 → 512 → 2p with GELU (unchanged width), split into two
  p-dimensional halves, one per output channel.
- Output: ρ̂ = ⟨b_k, τ_ρ(x, t_k)⟩ + b_ρ,  q̂ = ⟨b_k, τ_q(x, t_k)⟩ + b_q, p = 256
  per channel (the current 512 total).
- Optional physics regulariser (Phase 2, not in the core pipeline): the LWR
  residual ∂ρ/∂t + ∂q/∂x evaluated by autograd on the trunk coordinates,
  weighted small; the mesh-free trunk is what makes this possible and is one
  of the arguments for DeepONet over a grid model.

### 5.4 Loss, training, ensemble

- Loss per view: MSE on z-scored ρ over sampled query points (512 per
  rollout per epoch, as now) + λ_q · MSE on q̂(L, t_k) over all k + λ_band ·
  extra weight on cells with ρ > 40 veh/km upstream of the merge (the
  shockwave band), λ_q = 1, λ_band = 2, both tuned on validation.
- Optimiser: AdamW, lr 1e-3 with cosine decay, weight decay 1e-6, batch 16
  rollouts, gradient clip 1.0, 150 epochs, validation every 5, `best.pt` on
  validation relative-L2 (not MSE, so free-flow and jam rollouts count
  comparably).
- Ensemble: M = 5 members, seeds 0–4, each trained on a bootstrap resample of
  the training rollouts (63 % unique), same hyper-parameters. Saved as one
  directory with `member_{i}/best.pt` and a manifest; `SurrogateEnv` loads
  the manifest.
- Fine-tuning after aggregation: resume every member from its checkpoint,
  20 epochs on the enlarged set with lr 3e-4, keeping the original bootstrap
  mask plus all new rollouts.
- CPU note: M3 ran ~10 s/epoch on 1428 views; the round-0 set (Section 6)
  has ~4300 views, so ~30 s/epoch, ~75 min per member on CPU, five members
  in parallel processes. A GPU is convenient, not required.

### 5.5 Validation metrics (reported per regime, per family, per prefix)

| Metric | Why it matters for RL |
|---|---|
| relative L2 of ρ, split into free-flow / shockwave band / jam cells | where the error lives |
| relative L2 of q at the exit, and the episode-sum outflow error | the reward's delta term |
| per-prefix-length error (MLP branch) or per-k error (causal branch) | the RL-time input regime |
| **return-prediction error**: for each held-out rollout, the three-term return computed from the surrogate fields vs from SUMO labels, same reward weights | the quantity PPO actually optimises; the M6 §6.6 check generalised |
| breakdown-onset error (min) and false breakdown rate | the cliff |
| ensemble spread vs actual error (calibration plot) | whether disagreement is a usable uncertainty signal |
| constant-u and store-and-flush sweeps on the surrogate vs SUMO (per-term ranges) | reward-balance parity (Section 7.2) |

Acceptance gate for a surrogate to be used in RL (round 0): return-prediction
error ≤ 10 % on the validation rollouts of the behaviour mixture, false
breakdown rate ≤ 10 %, and ensemble calibration slope within [0.5, 2]. The
global relative-L2 number is reported but does not gate; the shockwave run's
0.235 shows that a global number over an adversarial family says little.

---

## 6. Data strategy

### 6.1 Round-0 dataset (N_0 = 480 rollouts ≈ 480 EE)

Every rollout: a random training-family profile, random SUMO seed,
`speed_dev` 0.03, one behaviour controller from the mixture below, occupancy
density, all labels of Section 4.3.

| Share | Behaviour controller | Purpose |
|---|---|---|
| 20 % | constant u on a 21-point grid (corners included) | anchors, capacity edge per profile |
| 25 % | ALINEA / PI-ALINEA with K_I ~ U[10, 40], ρ_set ~ U[30, 42], detector ∈ {11, 12, 13}, random u_init | closed-loop edge-riding, jam onset and recovery |
| 20 % | store-and-flush schedules: u_low ~ U[0, 0.3] during a window around the peak, u_high ~ U[0.5, 1] after, random switch times | the structure a good dynamic policy has |
| 15 % | run-7 policy (re-normalised to the new density estimator) with Gaussian action noise σ ∈ {0.05, 0.15} | the best known feedback policy, perturbed |
| 20 % | random piecewise-constant / smooth signals (existing families) | coverage of the rest of the input space |

Why 480: it is below the run-7 training budget (667 EE), enough for ~4300
training views with 8 padded views per rollout, and the M3 experience (120
rollouts → rel-L2 0.07 in free flow) suggests the free-flow part is easy;
the closed-loop share is what buys the jam regime. E1 measures the curve at
N ∈ {120, 240, 480, 960} so the paper can state the minimum.

Split 70 / 15 / 15 by rollout, stratified by controller type and by peak
total. Generation with the existing 8-worker launcher: ~480 × 20 s / 8 ≈
20–40 min.

### 6.2 Aggregation rounds

Round j (j = 1 … R, R = 4 by default):

1. PPO on the ensemble surrogate for S_j steps (S_j = 1M ≈ 8300 surrogate
   episodes ≈ 30 min batched; round 1 starts from `action_init_u`, later
   rounds warm-start from the previous round's selected checkpoint).
2. Rank checkpoints by the surrogate validation-set return (cheap), take the
   top 3.
3. Roll each of the 3 in SUMO on the 18 validation profiles, one seed
   (54 EE). This gives the transfer measurement for the round *and* the new
   data.
4. Append the 54 rollouts to the surrogate training set (they are labelled
   "on-policy, round j"); fine-tune the ensemble (Section 5.4).
5. Log: surrogate return vs SUMO return per checkpoint (the return-
   prediction error on the policy's own distribution), ensemble spread on
   those rollouts before and after fine-tuning, cumulative EE.

Stopping rule: the round-to-round improvement of the best SUMO validation
return is below 2 return units, or R rounds are done. Total budget for
R = 4: 480 + 4 × 54 = 696 EE, the same order as run 7's 667 EE, but the
sample-efficiency curve is read at every round (534, 588, 642, 696 EE) and
at round 0 (480 EE, zero-shot).

### 6.3 Optional SUMO fine-tuning (D8, C1)

Load the selected surrogate policy into PPO on SumoEnv (`PPO.load(path,
env=...)`, value network kept), lr 3e-5, `target_kl` 0.01, log-std reset to
−2, n_steps 480, B_ft ∈ {100, 200} EE, EvalCallback on V, multi-seed
selection as in run 7. This is the "pre-train then fine-tune" arm.

---

## 7. Surrogate environment and RL

### 7.1 SurrogateEnv v2

- Batched: a `VecEnv` implementation that holds n parallel episodes and runs
  one ensemble forward per step for all of them (branch batch n, trunk
  queries 19 points at t_k). n = 16 by default; SUMO cannot do this cheaply
  and it is one of the concrete benefits to report (episodes per minute).
- Ensemble handling: member index sampled per episode (default); options
  `mean` (average of members) and `pessimistic` (reward minus κ · ensemble
  std of the return terms, MOPO-style, κ swept in an ablation).
- Inputs per step: the env owns the profile (d_k, r_k), the analytic queue
  (release = min(Q + a_k, u_k · c)), and writes q_r,k = released / (c · Δt)
  into the branch history; the surrogate is never shown u.
- Outputs per step: ρ̂_k (19,), q̂_out,k; reward from `reward_terms` with
  outflow = q̂_out,k; clipping of ρ̂ to [0, 143 · lanes] and q̂ to [0, 3000].
- Observation and info identical to SumoEnv (Section 7.3), plus
  `ensemble_member`, `ensemble_std_density`, `ensemble_std_outflow`.
- Determinism: seeded member sampling; a `noise` option adds Gaussian noise
  scaled by the ensemble std to ρ̂ (ablation only).

### 7.2 Reward parity

Same `reward_terms`, same weights in both envs, with the offered-demand mode
(D10). Weights are re-balanced once on the profile family by running the E0
sweeps (constant u and store-and-flush) through SUMO *and* through the
round-0 surrogate; the balance script reports per-term ranges from both, and
the surrogate is accepted for RL only if each term's range agrees within
20 % (otherwise the surrogate distorts the trade-off the policy learns, and
that is a surrogate problem to fix, not a weight to tune). Warm-up mask
90 s, unchanged.

### 7.3 Observation (both envs)

    [ ρ_1..ρ_19 (z-scored, occupancy estimator) | d_k / 2500 | r_k / 1000 |
      look-ahead (optional): d_{k+1..k+20} / 2500, r_{k+1..k+20} / 1000 |
      k / K | Q_k / 100 ]

- Reactive variant: 23 features (as run 7, with the normalisers changed from
  min-max over levels to fixed constants so the layout does not depend on the
  training grid).
- Anticipative variant: + 40 look-ahead features (10 min). Justification:
  upstream detectors and ramp counters give a demand forecast with lead time
  in practice; the M7 analysis showed the RL margin over ALINEA comes from
  pre-emption, which a forecast should amplify. Both variants are trained on
  the surrogate; the SUMO arm trains only the reactive one unless budget
  allows (C5 shows the cost difference).
- Observation clipping: z-scores clipped to [−3, 25] before the tanh MLP
  (formulation.md §7.10).

### 7.4 PPO configuration (one file, `configs/rl/ppo_common.yaml`)

| Item | Value | Source |
|---|---|---|
| network | 3×512 tanh, separate pi / vf | run 7 |
| action | symmetric box, `action_init_u` 0.3, `log_std_init` −2 | run 7 |
| n_steps / batch / epochs | 480 / 120 / 5 (surrogate: 480 × 16 envs) | run 7, vectorised |
| lr, clip, target_kl, ent_coef | 1e-4, 0.2, 0.02, 0 | run 7 |
| gamma, lambda | 0.99, 0.95 | unchanged |
| eval | cycle the 18 validation profiles, deterministic, every 2400 steps (SUMO) / 24k steps (surrogate) | run 7, profiles instead of cells |
| checkpoints | every eval pass; post-hoc multi-seed selection (top-5, 18 profiles × 3 seeds, no episode < −150) | run 7 |
| seeds | 3 PPO seeds per arm (surrogate arm: 5, it is cheap) | new |

### 7.5 Checkpoint selection

Surrogate arm: rank by surrogate validation return, confirm top-3 in SUMO
(this is the aggregation rollout, so it costs nothing extra), pick by SUMO
validation return subject to the breakdown constraint. SUMO arm: as run 7.
Selection uses V only; T and O are touched once, at the end.

---

## 8. Transfer arms (what gets compared)

| Arm | SUMO budget (EE) | Description |
|---|---|---|
| A0 zero-shot | 480 | round-0 surrogate PPO, best checkpoint by SUMO validation (the 54 EE of validation rollouts are reported as selection cost) |
| A1 aggregation | 534 … 696 | Section 6.2, read at every round |
| A2 pre-train + fine-tune | 480 + 54 + {100, 200} | A0 policy fine-tuned in SUMO |
| B direct SUMO PPO | {200, 700, 2000} | the run-7 recipe on the profile family, budget-matched |
| ALINEA / PI-ALINEA | tuning cost (≈ 300 EE as in A.3) | tuned on V by the existing two-stage protocol |
| constant u | 21 EE | best single constant on V |
| Surrogate-MPC | 0 beyond the dataset | Section 9 |
| single-surrogate control | 480 | A0 with M = 1 deterministic member (C2) |

Every arm reports mean, 10th percentile, worst episode, breakdown rate,
recovery rate, TTS, served vehicles and final queue on T and O.

---

## 9. Baselines that the surrogate makes possible

**Surrogate-MPC.** At each step k, optimise u_{k..k+H−1} (H = 20 steps =
10 min) by gradient descent through the ensemble-mean DeepONet: the
inflow history up to k is known exactly, the candidate future controls
enter through the differentiable queue recursion (soft-min), the objective is
the same three-term reward summed over the horizon plus the terminal queue
term, 30 Adam iterations warm-started from the previous solution shifted by
one step; apply u_k. Offset-free correction: add (ρ_meas,k − ρ̂_k) to the
predicted densities over the horizon. Runtime ≈ 0.2 s per step, so it also
runs closed-loop inside SUMO. Why it belongs in the paper: it is the
classical model-based ramp-metering approach (METANET-MPC) with a learned
model, it needs no RL at all, and if it is competitive it changes the story
from "surrogate for RL" to "surrogate for control", which is a stronger
result either way. It is also a natural behaviour-cloning warm start for PPO
if round-0 zero-shot transfer is poor.

**One-step dynamics model.** MLP (or GRU) s_{k+1} = f(s_k, d_k, q_r,k) with
s = (ρ, q_out), trained on the same rollouts with one-step MSE, rolled
autoregressively in an otherwise identical env (same ensemble, same
aggregation). This is the control the reviewers will ask for; C4 is decided
by its long-horizon error and return-prediction error against the DeepONet.

---

## 10. Evaluation protocol and metrics

Primary metric: episode return under the training reward on T (mean over
90 episodes) with the breakdown constraint. Secondary, reported in the same
table: total time spent TTS = Δt Σ_k (Σ_i ρ_{i,k} Δx + Q_k + P_k) in veh·h
(P_k = pending mainline vehicles, the upstream queue), served vehicles,
final ramp queue, breakdown episodes (ρ_max > 60 veh/km for ≥ 5 min),
recovered breakdowns, worst episode, 10th percentile. TTS is the standard
objective in the ramp-metering literature and lets the paper be compared
with it; it is also computable on the surrogate directly, so a TTS-reward
ablation costs one surrogate PPO run (C5).

Figures:

1. Held-out return on T vs cumulative EE (log x), one curve per arm, ALINEA
   and constant as horizontal bands, 3-seed error bars. The headline.
2. Same with wall-clock on x.
3. Surrogate accuracy vs N_0 and vs data source (random vs behaviour mixture),
   with return-prediction error on the y-axis.
4. Transfer gap per round: surrogate-predicted vs SUMO return of the same
   checkpoints (scatter), before and after fine-tuning the ensemble.
5. Ensemble vs single surrogate: breakdown rate in SUMO of the policies each
   produces (C2).
6. Policy structure on a peaked profile: u(t), Q(t), d(t), r(t), density
   heat-map, for RL-reactive, RL-anticipative, ALINEA, Surrogate-MPC.
7. OOD family results.

Statistics: paired comparisons over the same 90 (profile, seed) episodes;
report mean difference with bootstrap 95 % intervals; three PPO seeds per
arm, five for the surrogate arm.

---

## 11. Experiment plan and gates

| ID | Experiment | Gate to proceed | SUMO cost |
|---|---|---|---|
| E0 | scenario characterisation (Section 4.4): insertion per block, capacity map, dynamic-vs-constant margin | dynamic schedule beats best constant by ≥ 15 return units on ≥ 2/3 of peaked profiles | ~150 EE, reused as data |
| E1 | surrogate study: N_0 ∈ {120, 240, 480, 960}; random-only vs behaviour mixture; branch A vs B; M = 1 vs 5; DeepONet vs one-step model | round-0 gate of Section 5.5 met at N_0 ≤ 480 with branch A | ≤ 960 EE once, shared |
| E2 | reward-parity check (Section 7.2) | per-term ranges within 20 % | 0 (uses E0 sweeps) |
| E3 | A0 zero-shot and A1 aggregation, 5 seeds | A1 at ≤ 700 EE ≥ direct PPO at 700 EE on V | ~220 EE per seed |
| E4 | B direct SUMO PPO at 200 / 700 / 2000 EE, 3 seeds | none (reference) | ~8700 EE total, the most expensive item; run in the background from day one |
| E5 | A2 fine-tune 100 / 200 EE, 3 seeds | A2 at 480 + 54 + 200 ≥ B at 2000 within 5 % | ~900 EE |
| E6 | baselines: ALINEA tuning on profiles, Surrogate-MPC, single-surrogate control | none | ~350 EE |
| E7 | ablations on the surrogate only: anticipative obs, TTS reward, terminal cost, pessimistic ensemble κ, delta = 0 (the old surrogate reward) | none | 0 (plus 90 EE each for the test-set evaluation of the winners) |
| E8 | final evaluation on T and O of every arm's selected policy | — | 90 + 36 EE per policy |

Dependencies: E0 → E1 → E2 → E3 → E5; E4 and E6 in parallel; E7 after E3;
E8 last.

---

## 12. Implementation plan

Ordered so that each step is testable on CPU without SUMO where possible.

**Step 1: profiles and labels (SUMO side).**
- `src/sumo_env/demand_profiles.py`: families of Section 4.1, seeded sampler,
  fixed V / T / O sets serialised to `configs/profiles/{val,test,ood}.json`.
- `network_builder._write_routes`: per-block flows; `MeteredRampQueue.step`
  takes `arrival_vph`.
- `run_simulation.py`: profile-driven mainline and ramp arrivals; controller
  mode (callable policy fed the SumoEnv-style observation); occupancy density
  with minGap correction and lane averaging; `outflow_vph`, `pending_mainline`
  labels; profile and controller metadata in the npz.
- `dataset_generation.py`: bring the family-schema generator into the tree
  (the untracked launcher imports `build_generation_plan`; reconcile with the
  version used for the shockwave set), add the behaviour-mixture plan of
  Section 6.1, stratified splits, an `--append-round j` mode.
- `SumoEnv`: profile sampling at reset (family or fixed set), per-step
  arrivals, occupancy density, fixed-constant demand normalisers, optional
  look-ahead features, observation clipping.
- Tests: profile sampler determinism, block flow XML, occupancy density unit
  test, queue with time-varying arrivals.

**Step 2: surrogate v2 (torch only).**
- `surrogate/deeponet.py`: `CausalBranch` (conv / GRU), two-channel trunk
  head, `DeepONetEnsemble` loader with manifest.
- `surrogate/datasets.py`: two-channel branch, outflow labels, band weights,
  causal mode (no padding) and padded mode, bootstrap masks.
- `surrogate/train.py`: relative-L2 validation, cosine schedule, `--member`
  and `--bootstrap-seed`, resume-fine-tune on an enlarged split.
- `surrogate/eval.py` + `scripts/eval_surrogate_regimes.py`: the metrics of
  Section 5.5 including return-prediction error and calibration.
- `scripts/train_ensemble.py`: launches M members as processes.
- Tests: causality (perturbing inputs after k leaves outputs at t_k
  unchanged, to 1e-6), shape contracts, ensemble loading.

**Step 3: SurrogateEnv v2 and RL.**
- `rl/surrogate_env.py`: batched `VecEnv`, profile sampling, inflow branch
  history, ensemble member sampling, outflow reward, parity info keys.
- `rl/reward.py`: `q_ref_mode: fixed | offered`, optional terminal cost.
- `rl/train_ppo.py`: `ppo_common.yaml` + env overlay; profile-cycling eval
  wrapper; `--init-policy` warm start for fine-tuning and aggregation rounds.
- `scripts/run_aggregation_loop.py`: the round loop of Section 6.2 with a
  ledger file (`runs/ledger/<study>.jsonl`, one line per SUMO rollout with
  its purpose).
- `scripts/balance_reward_terms.py`: accept surrogate sweeps and print the
  parity table.
- Tests: SurrogateEnv vs SumoEnv observation/info key parity, reward parity
  on synthetic inputs, vectorised env against the single env.

**Step 4: evaluation and baselines.**
- `scripts/eval_policy_profiles_sumo.py` (generalises `eval_policy_grid_sumo.py`
  to profile sets; returns TTS and the breakdown/recovery flags).
- `rl/baseline_controllers.py`: ALINEA on profiles (already
  observation-based, only the tuning script changes); `SurrogateMPC`
  controller with the offset correction.
- `surrogate/onestep.py`: the one-step baseline model and its env adapter.
- `scripts/plot_sample_efficiency.py`: figures 1–5 from the ledger and eval
  JSONL.

**Step 5: documentation.** `_plans/m8_*_plan.md` per step with the gates
above, `_progress/` twins, README pipeline section, and a proposal.md
revision (Section 13, needs approval).

Rough effort: step 1 ≈ 3–4 days, step 2 ≈ 3 days, step 3 ≈ 3 days, step 4 ≈
3 days, plus E4's SUMO time running in the background (~2 days of machine
time at 8 workers).

---

## 13. Risks and mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| The surrogate cannot represent the cliff even with ensembles (rel-L2 fine, breakdown onset wrong by > 5 min) | E1 onset error, false breakdown rate | causal branch + band weighting first; then a classifier head for "breakdown within 5 min" used as an extra reward penalty on the surrogate; then more closed-loop data near the edge (aggregation targets it automatically) |
| Zero-shot transfer is poor and aggregation is slow to fix it | round-1 SUMO validation far below surrogate prediction | warm-start PPO from the Surrogate-MPC or ALINEA by behaviour cloning; raise the per-round SUMO rollouts to 108 EE |
| Direct SUMO PPO on profiles also works well at 700 EE, erasing C1 | E4 | the claim then rests on C3 (anticipative variant), C4, C5 and wall-clock; the paper is still publishable as "when a learned plant model is and is not worth it" if the ledger is honest |
| Profile family too easy or too hard | E0 margin gate | adjust A, A_r caps; keep the family definition in one file so the change is one commit |
| SUMO backlog (pending mainline) after jams has no surrogate counterpart | recovery episodes mispredicted | keep pending vehicles out of the observation (parity), include recovery rollouts in the data (ALINEA and closure schedules produce them), report as a limitation |
| CPU-only machines | training time | ensemble members in parallel processes; batched surrogate env; the surrogate arm never needs a GPU, the SUMO arm never uses one |
| Run-7 checkpoint not on this machine | behaviour-mixture share | replace with ALINEA at two extra gain settings until it is copied over |

---

## 14. Milestones

| Milestone | Content | Exit criterion |
|---|---|---|
| M8 Scenario v2 | Step 1, E0 | E0 gate; V / T / O frozen and committed |
| M9 Plant model v2 | Step 2, E1, E2 | round-0 gate; parity table |
| M10 Surrogate RL | Step 3, E3 | A1 curve on V with 5 seeds; ledger complete |
| M11 Reference arms | E4, E5, E6 | all arms evaluated on V |
| M12 Study | E7, E8, figures, proposal/README update | paper-ready tables and figures |

---

## Appendix A. Notation (extends formulation.md)

| Symbol | Meaning |
|---|---|
| d_k, r_k | mainline demand and ramp arrival rate during interval k (vph), from the profile |
| q_r,k | released ramp inflow during interval k (vph) = min(Q_{k−1} + a_k, u_k D) · 3600/Δt |
| q_out,k | vehicles leaving the network in interval k × 3600/Δt |
| G_θ^{(m)} | ensemble member m of the plant model |
| EE | SUMO episode-equivalent, one 1-h rollout |
| V, T, O | validation, test, out-of-distribution profile sets |
| P_k | pending (blocked) mainline vehicles at the end of interval k |

## Appendix B. Config sketches

`configs/profiles/family_v1.yaml`

```yaml
mainline:
  families: {peak: 0.6, step: 0.3, const: 0.1}
  peak: {d_base: [1200, 1700], amplitude: [300, 900], t_center_min: [15, 35], half_width_min: [7.5, 15], d_max: 2300, peak_end_max_min: 45}
  step: {d_base: [1200, 1600], d_high: [1800, 2300], t1_min: [5, 25], t2_offset_min: [10, 20], p_step_down: 0.5}
  const: {d: [1500, 2000]}
ramp:
  families: {surge: 0.8, const: 0.2}
  surge: {r_base: [200, 500], amplitude: [0, 500], t_start_min: [5, 40], length_min: [10, 20], r_max: 1000}
  const: {r: [300, 800]}
block_min: 5
```

`configs/surrogate/plant_v2.yaml` (excerpt)

```yaml
model:
  branch: {type: causal_conv, channels: 128, kernel: 5, dilations: [1, 2, 4, 8], latent_dim: 256}
  trunk: {hidden_dim: 512, latent_dim: 256, outputs: [density, flow]}
data:
  branch_channels: [mainline_demand, ramp_inflow]
  normalisers: {mainline_demand: 2500, ramp_inflow: 1600, flow: 2500}
  band_weight: 2.0
  outflow_weight: 1.0
ensemble: {members: 5, bootstrap: true}
training: {n_epochs: 150, lr: 1e-3, schedule: cosine, batch_size: 16, select_on: val_rel_l2}
```

`configs/rl/env_surrogate.yaml` (overlay on `ppo_common.yaml`)

```yaml
env:
  type: surrogate
  ensemble_dir: runs/surrogate/plant_v2_round0
  ensemble_mode: sample        # sample | mean | pessimistic
  n_envs: 16
  profiles: configs/profiles/family_v1.yaml
  observation: {lookahead_steps: 0}   # 20 for the anticipative variant
  reward: {q_ref_mode: offered, q_cap: 2476, terminal_queue_weight: 0.0}
```

## Appendix C. Ledger line format

```json
{"study": "m10_seed0", "round": 1, "purpose": "aggregation", "profile_set": "val", "profile_id": 7, "sumo_seed": 10001, "policy": "ckpt_240k", "return": -71.2, "breakdown": false, "wall_s": 18.4}
```

Purposes: `dataset`, `aggregation`, `finetune`, `direct_ppo`, `tuning`,
`eval_val`, `eval_test`, `eval_ood`. Budget plots sum everything except
`eval_test` and `eval_ood`.

## Appendix D. Decisions needed from the user

1. Approve revising `proposal.md` (problem setting, surrogate I/O contract,
   reward section, evaluation) to this pipeline once M8 is under way; the
   current proposal still describes the M5c reward and a u-only branch.
2. Confirm the horizon stays at 1 h / K = 120 with the drain-tail rule, rather
   than a 90-min horizon.
3. Confirm that the 6000-rollout shockwave dataset is *not* the round-0 set
   (it is single-demand, action-driven, open-loop; it can serve E1's
   "random-only" data-source arm at 2000 vph if wanted).
4. Which machine runs E4 (the direct-SUMO reference at 2000 EE is the most
   expensive item, ~8–12 h at 8 workers), and whether a GPU is available for
   ensemble training (optional).
5. Whether run 7's `best_model_multiseed.zip` can be copied to this machine
   for the behaviour mixture.
