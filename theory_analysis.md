# Theory: when and why the plant-model surrogate pipeline pays

Status: analysis note for the paper (2026-09-13). Companion to
`formulation.md` (what the code computes) and `draft_pipeline.md` (the study
design). Every symbol below is one the pipeline already measures; §7 maps each
theoretical quantity to the number in `_progress/m9_*`, `m10_*`, `m12_*`, and
§8 says where the results go in the paper and what wording they support.
Nothing here changes the code. Numbers are the demo study (one PPO seed, CPU).

The question the note answers: in what sense is "PPO on a DeepONet plant model
plus budgeted aggregation" more efficient than "PPO in SUMO", and what can we
*prove* about it rather than only measure. The answer has four parts, each a
short result with the assumptions made explicit:

1. **Cost accounting** (Prop. 1): SUMO cost of the surrogate pipeline is fixed
   by identification, not by optimisation, so it is amortised over every PPO
   run, seed, ablation and observation variant; direct PPO pays per run.
2. **Transfer error is non-compounding** (Thm. 1, Prop. 2): because the
   DeepONet maps the *inflow history* to the field at $t_k$, the return error
   of a fixed control sequence is *linear* in the per-step field error, while a
   one-step autoregressive model compounds its error quadratically or
   exponentially in the horizon. This is the theoretical content of C4.
3. **Only two policies' model errors matter** (Thm. 3, Lemma 4): the policy
   optimised in the surrogate is near-optimal in SUMO up to the model error on
   *its own* trajectories and on the optimum's, not the worst case over all
   policies; the surrogate's checkpoint ranking is consistent under the same
   condition. This is why the round-0 mixture (D6) and aggregation (D7) are
   the right levers, and it is the theoretical content of C1.
4. **Aggregation controls the on-policy error** (Prop. 5): the loop is a
   DAgger-style reduction of model-based RL to no-regret online regression on
   the sequence of policy-induced distributions, so some round's policy is
   near-optimal up to the best-in-class model error.

Everything else (ensembles, universal approximation, the causal branch, the
batched environment) is a remark, not a theorem, and is labelled as such.

---

## 1. Setting and notation

One episode: horizon $K = 120$ control steps of $\Delta t = 30$ s, a demand
profile $\omega = (d_{0:K-1}, r_{0:K-1})$ drawn from the training family
$\mathcal{P}$, and SUMO noise $\xi$ (seed, `speed_dev` 0.03). Write
$\tau = \Delta t / 3600 = 1/120$ h.

**Meter (shared).** Given actions $u = (u_0, \dots, u_{K-1})$ and $\omega$, the
analytic queue returns the released inflow and the ramp queue,
$(q_{r}, Q) = \mathcal{A}(u, \omega)$, *identically* in both environments
(within the 1-vehicle integer rounding of SUMO's meter, absorbed into the
error terms below). The inflow history at step $k$ is
$h_k = (d_{0:k}, q_{r,0:k})$.

**Plant (SUMO).** By causality of the simulation
(`formulation.md` §2), the density field and the exit flow during interval
$k$ depend only on $h_k$ and $\xi$:

$$
\rho_k = G(h_k, \xi) \in \mathbb{R}^{N_x}, \qquad q_k = G^{q}(h_k, \xi) \in \mathbb{R}.
$$

**Surrogate (DeepONet ensemble member $m$).** $\hat\rho_k = \hat G_m(h_k)$,
$\hat q_k = \hat G^{q}_m(h_k)$: one forward pass of the causal branch on
$h_k$ and the trunk at the 19 detector positions. No predicted quantity is
fed back into the branch input (`surrogate_vec_env.py` §1: the history holds
$d$ and $q_r$ only).

**Reward (shared, `reward.form: tts`).**

$$
r_k = -\tau\,(N_k + Q_k + P_k), \qquad
N_k = \Delta x \sum_{i} \rho_{i,k}, \qquad
P_k = \max\!\big(0,\ \mathrm{Off}_k - \mathrm{Srv}_k - N_k - Q_k\big),
\qquad \mathrm{Srv}_k = \tau \sum_{j \le k} q_j,
$$

with $\mathrm{Off}_k$ the cumulative offered demand, a function of $\omega$
only. The episode return is $J = \sum_k r_k = -\mathrm{TTS}$ in veh h.

**Open-loop and closed-loop returns.** For a fixed action sequence $u$,
$J_M(u;\omega) = \mathbb{E}_\xi\, J$ and $J_{\hat M}(u;\omega)$ is the same
with $\hat G$ in place of $G$ (deterministic). For a policy $\pi$ mapping the
observation $o_k = (\rho_{k-1}, d_k, r_k, k/K, Q_{k-1})$ to $u_k$,
$J_M(\pi) = \mathbb{E}_{\omega,\xi} J$ and $J_{\hat M}(\pi) = \mathbb{E}_{\omega,m} J$
(member $m$ sampled per episode). $\pi^\star \in \arg\max_{\pi\in\Pi} J_M(\pi)$.

**Model error on a policy.**

$$
\varepsilon(\pi) := \big|\,J_M(\pi) - J_{\hat M}(\pi)\,\big| .
$$

This is the *transfer gap* the aggregation loop logs every round
(`m10_surrogate_rl_progress.md` §2, column "gap").

**Per-step field errors** for a given $(u, \omega)$:

$$
\varepsilon^{\rho}_k := \Delta x\; \mathbb{E}_\xi \|\rho_k - \hat\rho_k\|_1 \ \ [\text{veh}],
\qquad
\varepsilon^{q}_k := \mathbb{E}_\xi |q_k - \hat q_k| \ \ [\text{vph}] .
$$

**Costs.** $c_S$ = wall-clock of one SUMO episode, $c_{\hat S}$ = of one
surrogate episode, $c_{\rm fit}$ = ensemble training. $\mathrm{EE}$ = SUMO
episode equivalents as counted by the ledger.

---

## 2. Cost accounting: identification is paid once, optimisation is free

**Proposition 1 (budget decomposition and amortisation).** Let the surrogate
pipeline use a round-0 dataset of $N_0$ rollouts, $R$ aggregation rounds of
$n_{\rm agg}$ rollouts each, and let it run $J$ independent PPO jobs on the
surrogate (seeds, rounds, ablations, observation variants), each of
$S$ environment steps. Let the direct pipeline run $J'$ PPO jobs in SUMO of
$n_{\rm opt}$ episodes each plus $n_{\rm ev}$ evaluation episodes per job. Then

$$
\mathrm{EE}_{\rm surr} = N_0 + R\, n_{\rm agg} \quad (\text{independent of } J, S),
\qquad
\mathrm{EE}_{\rm direct} = J' (n_{\rm opt} + n_{\rm ev}),
$$

$$
W_{\rm surr} = (N_0 + R n_{\rm agg})\, c_S^{\rm par} + (R+1)\, c_{\rm fit} + J\, S\, c_{\hat S}/K,
\qquad
W_{\rm direct} = J' (n_{\rm opt} + n_{\rm ev})\, c_S^{\rm seq},
$$

where $c_S^{\rm par}$ is the cost of an *independent* SUMO episode (dataset
and aggregation rollouts are embarrassingly parallel) and $c_S^{\rm seq}$ the
cost of an episode generated *inside* on-policy PPO (one TraCI process per
learner, 4 episodes per update, every episode used once and discarded).

*Proof.* Counting; the only content is that surrogate episodes cost 0 EE and
that the two kinds of SUMO episode have different parallelism. $\square$

**Corollary 1 (break-even).** The surrogate pipeline uses fewer SUMO episodes
than the direct one as soon as $J' (n_{\rm opt} + n_{\rm ev}) > N_0 + R n_{\rm agg}$,
i.e. after the first direct run if $n_{\rm opt} + n_{\rm ev} > N_0 + R n_{\rm agg}$, and
otherwise after $\lceil (N_0 + R n_{\rm agg}) / (n_{\rm opt} + n_{\rm ev}) \rceil$
direct runs. Every additional surrogate-side experiment costs 0 EE and
$S c_{\hat S}/K$ wall-clock.

*Demo instantiation.* $N_0 = 692$, $R = 3$, $n_{\rm agg} = 54$: 854 EE bought
the three A1 rounds *and* the single-surrogate, one-step, anticipative and
Surrogate-MPC arms (five PPO jobs, $\approx 9$ min each). Direct PPO at
$n_{\rm opt} = 700$ cost 844 EE and 2.9 h for *one* job, i.e.
$c_S^{\rm seq} \approx 12.4$ s/EE against $c_S^{\rm par} \approx 1.7$ s/EE on
8 workers ($7$ s serial) and $c_{\hat S} \approx 0.22$ s per surrogate episode
(600 steps/s on 16 batched environments; the one-step plant runs at 4 800
steps/s). At paper scale (4 rounds of 1M steps, 3–5 seeds, three observation
variants) the surrogate side is $\approx 10^5$ episodes of optimisation at
0 EE; the same optimisation in SUMO would be $\approx 10^5$ EE.

*What the proposition does not say.* It does not say the surrogate policy is
any good. That needs §3–§5. It also does not say wall-clock is lower at demo
scale: A1's 3.9 h (E0 + dataset + ensemble + 3 rounds) exceeds the direct
arm's 2.9 h; the wall-clock advantage is in the marginal cost of the next run
(9 min vs 2.9 h) and in scaling $S$ (1M steps in 30 min vs $\approx 8\,300$
EE $\approx 29$ h in SUMO).

---

## 3. Transfer error of the operator surrogate is non-compounding

**Theorem 1 (open-loop return error, history-to-field surrogate).** For any
action sequence $u$ and profile $\omega$,

$$
\big|\, J_M(u;\omega) - J_{\hat M}(u;\omega)\, \big|
\;\le\; \tau \sum_{k=0}^{K-1} \Big( 2\,\varepsilon^{\rho}_k + \tau \sum_{j \le k} \varepsilon^{q}_j \Big)
\;\le\; 2\,\bar\varepsilon^{\rho}\,(K\tau) + \tfrac{1}{2}\,\bar\varepsilon^{q}\,(K\tau)^2 ,
$$

with $\bar\varepsilon^{\rho} = \max_k \varepsilon^{\rho}_k$,
$\bar\varepsilon^{q} = \max_k \varepsilon^{q}_k$. Since $K\tau = 1$ h: **a mean
absolute density error of $\bar\varepsilon^\rho$ vehicles on the road costs at
most $2\bar\varepsilon^\rho$ veh h of return, and a mean exit-flow error of
$\bar\varepsilon^q$ vph at most $\bar\varepsilon^q / 2$ veh h.** No factor grows
with $K$ beyond the horizon length itself.

*Proof.* $Q_k$ is identical in both environments ($\mathcal{A}$ is shared), so
$r_k - \hat r_k = -\tau\,[(N_k - \hat N_k) + (P_k - \hat P_k)]$. For the first
term, $|N_k - \hat N_k| = \Delta x\,|\sum_i (\rho_{i,k} - \hat\rho_{i,k})| \le
\Delta x \|\rho_k - \hat\rho_k\|_1$. For the second, $x \mapsto \max(0, x)$ is
1-Lipschitz, and $\mathrm{Off}_k$, $Q_k$ cancel, so
$|P_k - \hat P_k| \le |N_k - \hat N_k| + |\mathrm{Srv}_k - \widehat{\mathrm{Srv}}_k|
\le \Delta x\|\rho_k - \hat\rho_k\|_1 + \tau \sum_{j\le k} |q_j - \hat q_j|$.
Sum over $k$, take $\mathbb{E}_\xi$ (Jensen moves the expectation inside the
absolute values), and use $\sum_{k} \sum_{j \le k} 1 = K(K+1)/2 \le K^2$ for
$K\tau = 1$. $\square$

The $(K\tau)^2$ in the flow term is the integration of served vehicles over
the hour, not model compounding: each $\hat q_j$ is predicted from $h_j$
independently of $\hat q_{j-1}$. The key line of the proof is that
$\hat\rho_k$ and $\hat q_k$ enter once each; that is exactly D12 (keep the
history-to-field formulation) in theorem form.

**Proposition 2 (one-step autoregressive surrogate compounds).** Let the
alternative surrogate be $\hat s_{k+1} = \hat f(\hat s_k, d_k, q_{r,k})$ with
$s = (\rho, q)$, rolled from the empty state, with one-step error
$\varepsilon_1 = \sup_{(s, d, q_r) \in \mathcal{D}} \|\mathbb{E}_\xi f(s,d,q_r) - \hat f(s,d,q_r)\|_1$
on the data distribution $\mathcal{D}$ and Lipschitz constant $L$ of $\hat f$
in $s$. Then $e_k := \|s_k - \hat s_k\|_1$ satisfies
$e_k \le \varepsilon_1 \sum_{j<k} L^j$, and the return error is bounded by

$$
\big|J_M(u) - J_{\hat M}(u)\big| \;\le\; 3\tau \sum_k e_k \;\le\;
\begin{cases}
\tfrac32\,\varepsilon_1\, K(K+1)\,\tau = \tfrac32\,\varepsilon_1 (K+1) & L = 1,\\[2pt]
3\tau\,\varepsilon_1\, \dfrac{L^{K} - 1}{(L-1)^2} & L > 1,
\end{cases}
$$

i.e. $O(K^2 \varepsilon_1)$ at best and exponential in $K$ when $\hat f$ is
expansive. (Standard; e.g. Asadi, Misra & Littman 2018, Janner et al. 2019
Lemma B.3.)

*Proof.* Triangle inequality on $\hat s_{k+1} - s_{k+1} = [\hat f(\hat s_k) - \hat f(s_k)] + [\hat f(s_k) - f(s_k)]$, then Theorem 1's reward
argument with $e_k$ in place of both field errors: the density term gives
$2\tau \sum_k e_k$ and the cumulative flow term
$\tau^2 \sum_k \sum_{j \le k} e_j \le \tau^2 K \sum_j e_j = \tau \sum_j e_j$. $\square$

*Why $L \ge 1$ is the relevant regime here.* Near merge capacity the plant
is a knife-edge (M7, M8 §2: a release-rate change of one grid step flips the
episode between free flow and an irreversible jam), so the true one-step map
has a large local gain there and a fitted $\hat f$ inherits it. A one-step
model is therefore expansive exactly in the regime the policy has to operate
in. This is the theoretical reading of E1/C4: the one-step MLP on the same
692 rollouts diverges to $10^9$ on some rollouts unclipped; clipped to the
physical box its held-out field error is worse than the DeepONet's (M12 §4:
rel-L2 0.234 vs 0.177, jam cells 0.388 vs 0.208; the E1 table's
autoregressive number is in the thousands), and the policy it produces is
12 veh h worse on T (−81.9 vs −70.0). One honest caveat for the paper: its
*return*-prediction error on the mixture (0.094) is as good as the DeepONet's
(0.114), i.e. at 692 rollouts the compounding shows in the field and in the
transferred policy, not yet in the scalar return metric; the error-vs-horizon
curve of §7 is the measurement that separates the two cleanly.

**Remark 2.1 (closed loop: the one compounding mechanism that remains).**
Theorem 1 is open-loop. Under a policy, the actions in the two environments
diverge because the observations do, and this feeds back through the meter
into the inflows. Let $\pi$ be $L_\pi$-Lipschitz in the density part of the
observation (action units per vehicle), and let both plants have inflow
gain $S$: a released-inflow perturbation of $\delta$ vehicles at step $j$
changes $\|\rho_k\|_1 \Delta x$ by at most $S\delta$ for $k > j$. With
$D = 1600$ vph the meter converts an action deviation $e^u_j$ into at most
$D\tau e^u_j$ vehicles. Then the action deviation obeys
$e^u_k \le L_\pi\big(\varepsilon^\rho_{k-1} + S D \tau \sum_{j<k} e^u_j\big)$,
hence $e^u_k \le L_\pi \bar\varepsilon^\rho (1 + L_\pi S D \tau)^{k}$, and the
closed-loop return error is bounded by Theorem 1's bound plus
$2\tau S D \tau \sum_k \sum_{j<k} e^u_j$. Two things follow. (i) The factor
$(1 + L_\pi S D\tau)^K$ is shared by *every* surrogate, one-step models
included: they compound twice (through $\hat f$ and through $\pi$), the
operator surrogate once. (ii) The factor is controlled by the *policy's*
smoothness, which is why `target_kl`, the clipped observation and the
symmetric action box (D8) matter for transfer, not only for optimisation.
The demo's measured closed-loop gaps (+2.8, −0.2, +6.3 veh h, §7) are of the
same size as the open-loop return-prediction error (≈ 7 veh h), so at the
policies we actually obtain the closed-loop factor is near 1.

---

## 4. Only two policies' model errors matter

**Theorem 3 (transfer of the surrogate optimum).** Let
$\hat\pi \in \Pi$ satisfy $J_{\hat M}(\hat\pi) \ge \max_{\pi\in\Pi} J_{\hat M}(\pi) - \eta$
(PPO's optimisation gap on the surrogate). Then

$$
J_M(\hat\pi) \;\ge\; J_M(\pi^\star) - \varepsilon(\hat\pi) - \varepsilon(\pi^\star) - \eta .
$$

*Proof.* $J_M(\hat\pi) \ge J_{\hat M}(\hat\pi) - \varepsilon(\hat\pi)
\ge J_{\hat M}(\pi^\star) - \eta - \varepsilon(\hat\pi)
\ge J_M(\pi^\star) - \varepsilon(\pi^\star) - \varepsilon(\hat\pi) - \eta$. $\square$

The bound involves the model error on exactly two state distributions:
that of the policy the surrogate produces, $\varepsilon(\hat\pi)$, and that
of the true optimum, $\varepsilon(\pi^\star)$. Neither is the worst case
over $\Pi$ (which for a model trained on 692 rollouts would be useless).
The pipeline attacks each directly:

- $\varepsilon(\pi^\star)$ is what the **round-0 behaviour mixture (D6)** is
  for: store-and-flush schedules, ALINEA, and the run-7 policy are the
  families a good ramp-metering policy belongs to, so their rollouts put
  training data on or near $\pi^\star$'s state distribution. The per-regime
  return-prediction errors (5 % store-and-flush, 9–10 % constants and
  feedforward, 16 % ALINEA, 26 % run-7 policy; M9 §3) are the empirical
  proxy for $\varepsilon(\pi^\star)$ under different guesses of what
  $\pi^\star$ looks like.
- $\varepsilon(\hat\pi)$ is what **aggregation (D7)** reduces (§5).
- $\eta$ is paid in surrogate episodes at 0 EE, which is the whole point of
  Prop. 1: a large $S$ makes $\eta$ small for free.

**Lemma 4 (selection by surrogate return is consistent).** Let candidates
$\pi^{(1)}, \dots, \pi^{(n)}$ (the round's top checkpoints) satisfy
$\varepsilon(\pi^{(i)}) \le \bar\varepsilon$, and let $i^\star = \arg\max_i J_{\hat M}(\pi^{(i)})$.
Then $J_M(\pi^{(i^\star)}) \ge \max_i J_M(\pi^{(i)}) - 2\bar\varepsilon$.

*Proof.* Same three-line sandwich as Theorem 3. $\square$

So the surrogate's own ranking can be used to *choose which* checkpoints to
spend SUMO on (the pipeline confirms the top 3, not every checkpoint), and
ranking mistakes cost at most $2\bar\varepsilon$. In the demo the surrogate
top-1 was the SUMO top-1 in 3/3 rounds.

**Corollary 3.1 (what a "good enough" surrogate means).** To guarantee
$J_M(\hat\pi) \ge J_M(\pi^\star) - \delta$ it suffices that
$\varepsilon(\hat\pi) + \varepsilon(\pi^\star) \le \delta - \eta$. With
Theorem 1, a sufficient condition in field terms is
$2\bar\varepsilon^\rho + \tfrac12 \bar\varepsilon^q \le (\delta - \eta)/2$ on
the two distributions. This is the justification for the round-0 gate of
`draft_pipeline.md` §5.5 (return-prediction error ≤ 10 % on the mixture): it
is a bound on $\varepsilon(\cdot)$ over the mixture's policies, i.e. on
$\varepsilon(\pi^\star)$ if $\pi^\star$ resembles the mixture.

---

## 5. Aggregation controls the on-policy error

Theorem 3 leaves $\varepsilon(\hat\pi)$ open: the surrogate is fitted before
$\hat\pi$ exists, and $\hat\pi$ is drawn to wherever the model is
optimistic (the classic model-exploitation failure). The aggregation loop is
the standard remedy, and it admits the DAgger-style guarantee.

**Setting.** Round $j = 1, \dots, R$: fit $\hat G_{j-1}$ on
$\mathcal{D}_{j-1}$; optimise $\pi_j$ on $\hat G_{j-1}$ (gap $\eta$); roll
$\pi_j$ in SUMO for $n_{\rm agg}$ episodes; $\mathcal{D}_j = \mathcal{D}_{j-1} \cup$
those rollouts. Let $\ell_j(\hat G) := \varepsilon_{\pi_j}(\hat G)$ be the
on-policy model error of $\hat G$ under $\pi_j$ (a bounded loss, since
returns are bounded), and let
$\varepsilon_\star := \min_{\hat G \in \mathcal{H}} \frac1R \sum_j \ell_j(\hat G)$
be the best-in-class average error over the visited distributions.

**Proposition 5 (aggregation as no-regret model fitting).** Suppose the
fitting procedure is no-regret on the sequence $\ell_1, \dots, \ell_R$:
$\frac1R \sum_j \ell_j(\hat G_{j-1}) \le \varepsilon_\star + \gamma_R$ with
$\gamma_R \to 0$. (Follow-the-leader on the aggregated set, which is what
"fine-tune on $\mathcal{D}_j$" approximates, is no-regret for strongly convex
losses such as squared error over a fixed feature class; for a neural
network this is an assumption, made explicit.) Then there is a round
$j^\dagger \le R$ with

$$
J_M(\pi_{j^\dagger}) \;\ge\; J_M(\pi^\star) - \varepsilon(\pi^\star) - \varepsilon_\star - \gamma_R - \eta ,
$$

and the same holds for the round that the SUMO validation return selects
(up to the validation noise), since selection on V is what the loop does.

*Proof.* Theorem 3 at round $j$ gives
$J_M(\pi_j) \ge J_M(\pi^\star) - \varepsilon(\pi^\star) - \ell_j(\hat G_{j-1}) - \eta$,
and the average of $\ell_j(\hat G_{j-1})$ is at most
$\varepsilon_\star + \gamma_R$, so its minimum over $j$ is too. $\square$

(Ross, Gordon & Bagnell 2011 give the argument for imitation; Ross & Bagnell
2012, "Agnostic system identification for model-based RL", give the
model-based version this follows.)

**What the proposition buys.** The on-policy error term $\varepsilon(\hat\pi)$
of Theorem 3 is replaced by the *best-in-class* error $\varepsilon_\star$ over
the visited distributions plus a vanishing regret term. The model's
off-distribution optimism is turned, round by round, into data. Each round
costs $n_{\rm agg}$ EE, and $n_{\rm agg} = 54$ is also the number of SUMO
episodes the transfer measurement needs anyway, so aggregation is free in
EE relative to a pipeline that merely validates.

**Empirical face.** Return-prediction error of the ensemble on the new
on-policy rollouts, before → after the round's fine-tune:
0.185 → 0.128, 0.101 → 0.099, 0.101 → 0.079 (M10 §2). The sequence of
"before" values is the $\ell_j(\hat G_{j-1})$ of the proposition; it fell
from 0.185 (round-0 model on round-1 data) to 0.101 and stayed there, while
the policy went from −68.5 to −49.5 on V. A2 (100 EE of SUMO PPO on the
round-1 policy) at −64.6 vs A1's −58.4 / −49.5 with the same 54–108 EE spent
on aggregation is the demo's evidence that, at this budget, SUMO episodes are
worth more as constraints on the model than as policy-gradient samples.

**Remark 5.1 (why identification data is richer than policy-gradient data).**
The same SUMO episode contributes (a) $K N_x + K = 2\,400$ labelled scalars to
the regression, all of which constrain $\hat G$ for *every* future policy and
gradient step, or (b) $K = 120$ transitions to one PPO update, used for five
epochs and discarded (on-policy). This is not a theorem about rates (PPO has
no tight finite-sample bound, and the regression rate depends on the class),
but it is the mechanism behind Prop. 1's $J$-independence and behind the A1
vs A2 comparison. It can be made quantitative in one respect: with $n_{\rm env}$
batched surrogate environments the PPO batch is $n_{\rm env} \times$ larger at
the same wall-clock, so the advantage-estimate variance per update is
$n_{\rm env}$ times smaller (16 in `env_surrogate.yaml`) than the direct
arm's 4-episode batch.

---

## 6. Remarks that are not theorems (and how to phrase them)

**R1. Ensemble member sampling.** With a member drawn per episode PPO
maximises $\frac1M \sum_m J_{\hat M_m}(\pi)$, and Theorem 3 holds with
$\varepsilon(\pi)$ replaced by $\frac1M \sum_m \varepsilon_m(\pi)$. The
ensemble's spread is a proxy for that error (calibration slope 1.5–1.8, M9
§3), and the pessimistic mode (reward minus $\kappa\,\cdot$ spread) is a
lower confidence bound when the spread is calibrated (the MOPO argument). The
demo does **not** support the stronger claim that member sampling improves
zero-shot transfer (C2: single-member policy −72.3 vs ensemble −70.0 on T,
0 % vs 13 % breakdowns). In the paper the ensemble is presented as (i) the
uncertainty signal that drives aggregation and selection and (ii) a
robustness option, not as a proven transfer gain.

**R2. Realisability and causality.** The regression target
$h \mapsto \mathbb{E}_\xi\, G(h, \xi)$ is a well-defined causal operator on the
compact input set $[0, 2300] \times [0, 1600]$ per step, so the DeepONet
universal-approximation theorem (Chen & Chen 1995; Lu et al. 2021) applies to
it. Restricting the branch to causal encoders (GRU / causal conv, D4) removes
no realisable operator (the true one is causal) and removes the padded-MLP's
freedom to depend on future slots. E1: padded MLP rel-L2 0.329 vs GRU 0.177 on
the same data. This is an inductive-bias / sample-efficiency argument, stated
as such.

**R3. Stochastic plant.** $\hat G$ regresses to the conditional mean over
$\xi$; the reward is affine in $\rho$ and $q$ except through the clipped
backlog, so the return of the mean field equals the mean return up to the
backlog's clipping (Jensen gap), which is why the return-prediction error is
a fair metric and why Theorem 1 uses $\mathbb{E}_\xi$ of the absolute error,
not the error of the mean.

**R4. Mesh-free trunk.** The trunk lets the same model be queried at any
$(x, t)$, so a detector layout change, a different reward integration grid or
the LWR residual for PI-DeepONet cost no retraining. Not an efficiency
result; one sentence in the method section.

---

## 7. Instantiating the bounds with measured quantities

| symbol | what measures it | demo value | source |
|---|---|---|---|
| $c_S^{\rm par}$ | independent SUMO episode, 8 workers | 1.7 s (7 s serial) | M8 §0 |
| $c_S^{\rm seq}$ | SUMO episode inside on-policy PPO | 12.4 s (844 EE in 2.9 h) | M11 §2 |
| $c_{\hat S}$ | surrogate episode, 16 batched envs | 0.22 s (600 steps/s) | M10 §0 |
| $c_{\rm fit}$ | 5-member ensemble, 300 epochs | 31 min; fine-tune 2–3 min | M9 §2, M10 |
| $N_0,\ R,\ n_{\rm agg}$ | ledger | 692, 3, 54 | M10 §3 |
| $J S / K$ | surrogate episodes of optimisation | 7 500 (3 rounds); $\approx 33\,000$ at paper scale | M10 §0 |
| field error (rel-L2 $\rho$, val) | E1 / round-0 gate | 0.177 (jam cells 0.208) | M9 §3 |
| return-prediction error (val / test) | held-out $\lvert J_M - J_{\hat M}\rvert / \lvert J_M\rvert$, mixture policies | 0.114 / 0.090 | M9 §3 |
| $\varepsilon(\pi)$ per regime (proxy for $\varepsilon(\pi^\star)$) | return error by behaviour family | 0.05 (store-and-flush) … 0.26 (run-7) | M9 §3 |
| $\varepsilon(\pi_j)$, on-policy | transfer gap of the selected checkpoint | +2.8, −0.2, +6.3 veh h on $\lvert J\rvert \approx 50$–70 | M10 §2 |
| $\ell_j(\hat G_{j-1})$ | return error on round-$j$ rollouts before fine-tune | 0.185, 0.101, 0.101 | M10 §2 |
| Lemma 4 | surrogate top-1 = SUMO top-1 | 3 / 3 rounds | M10 §2 |
| Prop. 2 | one-step model field error / policy | diverges unclipped; −81.9 vs −70.0 on T | M12 §4–5 |
| Theorem 3 outcome | A1 vs direct at matched EE | −54.7 vs −64.1 on T (854 vs 844 EE); −70.4 vs −85.0 on O | M12 §2–4 |

**Consistency check of Theorem 1 (worth one sentence in the paper).** The
train-split mean density is 20.9 veh/km over 1.9 km, i.e. ≈ 40 vehicles on
the road on average; a relative field error of 0.18 is then
$\bar\varepsilon^\rho \approx 7$ veh and Theorem 1 bounds the return error by
≈ 14 veh h plus the flow term (rel-L2 0.053 on ≈ 2 000 vph ≈ 100 vph →
≈ 50 veh h worst case, but the exit-flow error is zero-mean over an episode by
conservation, so its cumulative term is far below the bound). The measured
return-prediction error is 0.114 × |J| ≈ 60 veh h ≈ 7 veh h. The bound is
therefore loose by a factor of ≈ 2 on the density term and the observed
error sits inside it; the flow term's looseness is the sign-cancellation
the $L_1$ bound cannot see. The closed-loop gaps (+2.8 / −0.2 / +6.3 veh h)
are of the same order, i.e. Remark 2.1's compounding factor is ≈ 1 at the
policies obtained.

**What is missing to make §7 tight (cheap, no new SUMO except the last).**

1. *Error-vs-horizon curve* for the one-step model vs the DeepONet: $k$-step
   ahead field error on the validation rollouts, $k = 1 \dots 120$. Prop. 2
   predicts growth, Theorem 1 predicts a flat profile. This is the standard
   figure for the compounding claim and costs 0 EE.
2. *Direct measurement of $\bar\varepsilon^\rho_k$ and $\bar\varepsilon^q_k$* in
   vehicles and vph per step (the evaluation script has the fields; only the
   units change) so the Theorem 1 bound can be drawn next to the measured
   return error per rollout (a scatter with the bound as a line).
3. *Per-round transfer-gap scatter* (Fig. 4) with the Theorem 1 bound of each
   checkpoint's own rollouts: this shows $\varepsilon(\pi_j)$ against its
   bound.
4. *$\varepsilon(N)$* at $N_0 \in \{120, 240, 486, 692\}$ with the full
   5 × 300-epoch recipe: the E1 curve (0.145 / 0.246 / 0.114) is noisy at
   3 × 100 epochs and the paper needs the monotone version to state
   $n_{\rm id}(\delta)$.
5. *The 2000-EE direct point* (E4), to read $n_{\rm opt}(\delta)$ off the direct
   curve at the return A1 reaches; this is the only item that costs SUMO.
6. Optional: a finite-difference estimate of $L_\pi$ (action change per
   vehicle of density perturbation) for the selected policies, to show
   Remark 2.1's factor is near 1.

---

## 8. How this goes into the paper

**Placement.** One section, "Analysis: when does a plant-model surrogate pay?",
between the method (pipeline) and the experiments, 1–1.5 pages: Prop. 1 in a
paragraph, Theorem 1 with a two-line proof sketch, Prop. 2 as a contrast
statement, Theorem 3 with its three-line proof, Prop. 5 as a statement with
the assumption named; full proofs and Remark 2.1 in an appendix. The
experiments section then reads each result off a figure, which is the
strongest form for a systems paper: the theory says which quantities matter
and the pipeline measures those quantities.

**Contribution sentence.** "We give a transfer bound for history-to-field
(operator) surrogates in which the return error is linear in the per-step
field error, in contrast to the quadratic-to-exponential compounding of
one-step dynamics models, and show that only the model error on two
policy-induced distributions, the surrogate optimum's and the true optimum's,
controls policy suboptimality. The round-0 behaviour mixture and the
budgeted aggregation loop target exactly these two quantities, and every
term in the bounds is measured in the study."

**Mapping to claims and figures.**

| result | claim | figure / table |
|---|---|---|
| Prop. 1 | C5 amortisation; C1 budget-matched comparison | Fig. 1 (return vs EE), Fig. 2 (vs wall-clock), arms table |
| Theorem 1 + check | surrogate fidelity is sufficient in return terms; the gate is principled | Fig. 3 (return error vs $N_0$), Fig. 4 (transfer gap with bound), E2 parity table |
| Prop. 2 | C4 operator vs one-step | error-vs-horizon figure (new, item 1 of §7), E1 table, one-step arm on T / O |
| Theorem 3 + Lemma 4 | C1; why selection by surrogate return is safe | per-round top-1 agreement, A0 zero-shot vs direct at 290 EE |
| Prop. 5 | C1 aggregation; A1 vs A2 | per-round return error before / after fine-tune, A1 curve, A2 point |
| R1 | C2 (negative at demo scale, stated honestly) | Fig. 5 breakdown rates, single-surrogate arm |
| R2 | causal branch as inductive bias | E1 branch ablation |

**Reviewer questions the section pre-empts.**

- *"Why not just run SUMO in parallel?"* Prop. 1: parallel SUMO lowers
  $c_S^{\rm par}$ for the dataset (we use it), but on-policy PPO's episodes
  are sequential with the learner ($c_S^{\rm seq}$ 7× higher in the demo) and
  are paid per run; the surrogate's SUMO cost is independent of the number of
  runs, seeds, steps and ablations.
- *"A learned model cannot be trusted off-distribution."* Theorem 3: only
  two distributions matter, and Prop. 5 shows the loop drives the on-policy
  one to best-in-class error at 54 EE per round; the measured on-policy gaps
  are 0–6 veh h.
- *"Why a DeepONet rather than an MLP dynamics model?"* Prop. 2 vs Theorem 1,
  with the horizon curve and C4.
- *"Is the ensemble necessary?"* R1: presented as the uncertainty signal for
  the loop and as an option, with the negative zero-shot result reported.
- *"Is the bound vacuous?"* §7's check: loose by ≈ 2 on the density term,
  observed errors inside it.

**Assumptions to state in one place (the paper's honesty paragraph).**
Empty initial road and a shared analytic meter (so the inflow history is a
sufficient input; a non-empty initial state would need a state input to the
branch); deterministic policies (Theorem 3 and Lemma 4 extend to stochastic
ones with $\varepsilon$ defined on the induced distributions); the no-regret
assumption of Prop. 5 for a fine-tuned network; PPO's optimisation gap
$\eta$ treated as an empirical quantity; all bounds worst-case in $L_1$, hence
sign-cancellation makes them loose, which the check in §7 quantifies.

**Changes this implies for the documents (need approval for the first).**

- `proposal.md`: add the analysis as a stated deliverable (section "Analysis")
  and C4's wording ("non-compounding return error") to the research
  questions.
- `_plans/m12_study_plan.md` / `_progress/m12_study_progress.md`: add §7's
  items 1–4 as E7-type (0 EE) tasks and item 5 as the already-planned E4
  paper-scale point.
- `README.md`: pointer to this note next to `formulation.md`.
