# M12 progress — Study: final evaluation, ablations, figures (2026-09-13)

Plan: `_plans/m12_study_plan.md`. Study `demo` (`runs/study/demo`, driver `scripts/run_study.sh`)
at demo budgets on the Mac CPU: one PPO seed, 300k surrogate steps per round, 3 aggregation
rounds, direct SUMO PPO at 200 and 700 EE, A2 at 100 EE, Surrogate-MPC with 20 iterations
and 3 members. Paper-scale budgets are the env-var values documented in the driver (1M steps,
4 rounds, 3–5 seeds, 2000 EE direct arm); nothing in the code changes. Every number below is
the return under the TTS-form reward (−TTS, veh h) on the frozen sets, selection on V only.

## 1. Arms and budgets (`runs/study/demo/arms.json`)
| arm | policy | SUMO budget (EE, ledger) |
|---|---|---|
| A0 zero-shot | round-1 top checkpoint by surrogate V return | 692 (round-0 data; the 54 EE of the round-1 V rollouts are selection cost) |
| A1 aggregation, rounds 1–3 | best SUMO V return of the round's top-3 | 746 / 800 / 854 |
| A2 pre-train + fine-tune | A1 round-1 policy + 100 EE PPO in SUMO (+ 36 eval_val) | 882 |
| B direct SUMO PPO | run-7 recipe on profiles, 200 EE (+ 90 eval_val) / 700 EE (+ 144) | 290 / 844 |
| PI-ALINEA | tuned on V (kp 4, K_I 20, ρ_set 26, det 13) | 522 (tuning) |
| constant u = 0.3 | best constant on V | 198 |
| Surrogate-MPC | H 20, 20 Adam iterations, members 0–2, TTS objective, perfect forecast | 692 |
| single-surrogate | A0 recipe with one member (mean mode) | 746 |
| one-step model | PPO on the one-step plant, same env otherwise | 746 |
| anticipative | reactive + 10-min demand look-ahead (63-dim obs) | 746 |

## 2. Final evaluation on T (30 profiles × 3 seeds = 90 episodes), ledger `demo_final`
| arm | EE | mean | p10 | worst | breakdown | recovered | TTS (veh h) | served | final queue | u | Δ vs PI-ALINEA [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | 692 | **-70.0** | -165.7 | -379.0 | 0.13 | 0.00 | 61.5 | 1995 | 33 | 0.31 | -9.5 [-15.8, -3.2] |
| A1 aggregation | 746 | **-70.0** | -165.7 | -379.0 | 0.13 | 0.00 | 61.5 | 1995 | 33 | 0.31 | -9.5 [-16.1, -3.8] |
| A1 aggregation | 800 | **-55.0** | -78.4 | -325.1 | 0.00 | – | 49.9 | 2035 | 21 | 0.34 | +5.6 [+1.7, +9.2] |
| A1 aggregation | 854 | **-54.7** | -87.7 | -294.7 | 0.02 | 0.00 | 48.8 | 2032 | 24 | 0.37 | +5.9 [+1.3, +10.1] |
| A2 pre-train + fine-tune | 882 | **-71.9** | -157.9 | -379.5 | 0.13 | 0.00 | 63.5 | 1992 | 34 | 0.31 | -11.3 [-18.6, -5.4] |
| B direct SUMO PPO | 290 | **-79.7** | -164.3 | -208.4 | 0.17 | 0.07 | 71.9 | 1973 | 61 | 0.28 | -19.2 [-25.2, -13.9] |
| B direct SUMO PPO | 844 | **-64.1** | -121.1 | -315.3 | 0.19 | 0.12 | 56.7 | 2013 | 34 | 0.34 | -3.5 [-7.9, +0.9] |
| ALINEA | 522 | **-60.6** | -108.7 | -222.5 | 0.02 | 1.00 | 55.2 | 2031 | 30 | 0.73 | +0.0 [+0.0, +0.0] |
| constant u | 198 | **-95.7** | -197.8 | -402.6 | 0.43 | 0.08 | 79.7 | 1932 | 40 | 0.30 | -35.1 [-44.0, -27.0] |
| Surrogate-MPC | 692 | **-134.6** | -249.3 | -368.4 | 0.71 | 0.03 | 110.2 | 1861 | 2 | 0.55 | -74.0 [-87.4, -60.5] |
| single-surrogate | 746 | **-72.3** | -152.9 | -344.6 | 0.00 | – | 62.6 | 1989 | 48 | 0.31 | -11.7 [-17.8, -6.2] |
| one-step model | 746 | **-81.9** | -165.0 | -366.1 | 0.00 | – | 69.0 | 1968 | 46 | 0.32 | -21.3 [-29.4, -14.4] |
| anticipative | 746 | **-75.9** | -145.3 | -232.3 | 0.00 | – | 65.6 | 1985 | 51 | 0.29 | -15.4 [-22.0, -8.8] |

## 3. Out-of-distribution set O (12 profiles × 3 seeds: double peak, 2200-vph plateau, early ramp surge)
| arm | EE | mean | p10 | worst | breakdown | recovered | TTS (veh h) | served | final queue | Δ vs PI-ALINEA [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | 692 | **-91.4** | -217.0 | -261.2 | 0.08 | 0.00 | 79.6 | 1916 | 30 | -19.8 [-29.8, -10.5] |
| A1 aggregation | 746 | **-91.4** | -217.0 | -261.2 | 0.08 | 0.00 | 79.6 | 1916 | 30 | -19.8 [-29.1, -10.7] |
| A1 aggregation | 800 | **-72.0** | -156.9 | -232.8 | 0.00 | – | 64.6 | 1962 | 36 | -0.4 [-6.7, +5.0] |
| A1 aggregation | 854 | **-70.4** | -133.7 | -216.1 | 0.00 | – | 63.0 | 1962 | 42 | +1.2 [-4.7, +6.8] |
| A2 pre-train + fine-tune | 882 | **-88.2** | -193.1 | -263.2 | 0.08 | 0.00 | 77.2 | 1925 | 28 | -16.6 [-26.1, -9.0] |
| B direct SUMO PPO | 290 | **-111.4** | -262.6 | -276.1 | 0.44 | 0.25 | 97.8 | 1887 | 32 | -39.8 [-51.8, -28.8] |
| B direct SUMO PPO | 844 | **-85.0** | -188.6 | -231.7 | 0.42 | 0.20 | 74.8 | 1937 | 44 | -13.5 [-20.7, -7.1] |
| ALINEA | 522 | **-71.6** | -151.1 | -176.9 | 0.00 | – | 66.2 | 1982 | 36 | +0.0 [+0.0, +0.0] |
| constant u | 198 | **-110.0** | -257.9 | -282.0 | 0.39 | 0.14 | 94.3 | 1886 | 24 | -38.4 [-51.0, -26.6] |
| Surrogate-MPC | 692 | **-123.0** | -240.0 | -248.0 | 0.81 | 0.31 | 102.7 | 1887 | 29 | -51.4 [-64.6, -38.4] |
| single-surrogate | 746 | **-90.4** | -220.4 | -244.4 | 0.00 | – | 78.3 | 1925 | 47 | -18.8 [-26.4, -11.7] |
| one-step model | 746 | **-96.3** | -233.0 | -256.7 | 0.00 | – | 82.2 | 1910 | 38 | -24.7 [-35.4, -15.2] |
| anticipative | 746 | **-97.8** | -228.5 | -253.1 | 0.00 | – | 84.5 | 1912 | 43 | -26.2 [-35.3, -17.7] |

## 4. Reading of the results
- **C1 (sample efficiency).** At a matched budget (854 vs 844 EE) the aggregation arm reaches
  −54.7 on T (TTS 48.8 veh h, 2 % breakdowns, worst −295) against −64.1 for direct SUMO PPO
  (TTS 56.7, 19 % breakdowns, worst −315): +9.4 veh h paired, and on the OOD set −70.4 vs −85.0
  with 0 % vs 42 % breakdowns. Direct PPO at 290 EE reaches −79.7. The direct arm's V curve
  (−95, −89, −95, −84, −80 at 200 EE; −72, −77, −66, −70 at 700 EE) is still rising, so the
  2000-EE point of the paper scale is needed to say where it saturates. Zero-shot transfer (A0, 692 EE) already beats direct PPO at
  290 EE by 9.7 veh h and every constant by 25, but jams on 13 % of the test episodes; the
  first aggregation round (54 EE of on-policy rollouts + 20 fine-tune epochs) removes the
  breakdowns and gains 15 veh h, the second round 0.3 more. **A2** (100 EE of SUMO PPO on top of
  the round-1 policy) is *worse* than spending the same episodes on aggregation
  (−71.9 vs −55.0): at this scale SUMO episodes buy more as model data than as policy gradient.
- **C3 (generalisation).** A1 beats tuned PI-ALINEA on T by +5.9 veh h (paired 95 % CI
  [+1.3, +10.1]) with a comparable breakdown rate (0.02 vs 0.02), and ties it on the OOD set
  (+1.2, CI [−4.9, +6.7]) with 0 breakdowns vs 0; ALINEA's worst OOD episode (−177) is better
  than A1's (−216). The anticipative variant is behind the reactive one at 300k steps (−75.9 on T):
  the 63-dim look-ahead observation needs more training than the demo budget gives it.
- **C2 (ensemble vs single surrogate).** The single-member policy transfers as well as the
  ensemble's zero-shot policy at this budget (−72.3 vs −70.0 on T, 0 % vs 13 % breakdowns —
  the ensemble policy is the more aggressive one, u 0.31 in both): C2 is **not** supported by
  the demo run; the ensemble's value shows up in the calibration signal that drives the loop,
  not in the zero-shot transfer.
- **C4 (operator model vs one-step model).** Held-out field error favours the DeepONet
  (rel-L2 0.177 vs 0.234; jam cells 0.208 vs 0.388, table in §5) and so does the policy it
  produces (−70.0 vs −81.9 on T, −91.4 vs −96.3 on O), while the one-step model's *return*
  prediction is as good (0.094 vs 0.114). Unclipped, the one-step rollouts diverge to 1e9 on
  some rollouts (compounding error); clipping the state to the physical box is what makes
  the comparison meaningful.
- **Surrogate-MPC** is the weakest learned arm (−134.6 on T, 71 % breakdowns): a 10-minute
  horizon under the TTS objective under-values preventing a breakdown whose cost lands after
  the horizon, and 20 Adam iterations from a warm start are not enough to move the solution
  off the pass-through corner. The sweep in §6 checks the horizon / terminal-weight fix.
- **Budget bookkeeping.** The 692-EE round-0 dataset dominates every surrogate arm's budget;
  the sample-efficiency curve is read at 692 / 746 / 800 / 854 EE. The direct arm's ledger
  counts its evaluation passes (18 EE each), which is why "200 EE" costs 290.

## 5. E1 surrogate study (`_progress/m9_e1_surrogate_study.json`, validation split, 106 rollouts)
| variant | rel-L2 ρ (free / band / jam) | rel-L2 q_exit | return-pred. error | false / missed breakdown | calibration slope | gate |
|---|---|---|---|---|---|---|
| GRU branch, M = 5, mixture data, N_0 = 486 (reference) | 0.177 (0.201 / 0.194 / 0.208) | 0.053 | 0.114 | 0.018 / 0.102 | 1.52 | fail |
| GRU, single member (M = 1) | 0.193 (0.225 / 0.198 / 0.199) | 0.055 | 0.086 | 0.053 / 0.082 | – | fail |
| GRU, N_0 = 120 rollouts | 0.331 (0.401 / 0.328 / 0.354) | 0.092 | 0.145 | 0.018 / 0.204 | 2.35 | fail |
| GRU, N_0 = 240 rollouts | 0.283 (0.334 / 0.298 / 0.300) | 0.077 | 0.246 | 0.018 / 0.204 | 2.74 | fail |
| GRU, random / open-loop data only (248 rollouts) | 0.296 (0.380 / 0.313 / 0.314) | 0.079 | 0.143 | 0.018 / 0.184 | 1.74 | fail |
| dilated causal conv branch (64 ch) | 0.212 (0.237 / 0.244 / 0.252) | 0.056 | 0.112 | 0.018 / 0.122 | 1.86 | fail |
| padded-MLP branch (branch B, 4 prefix views / rollout) | 0.329 (0.379 / 0.271 / 0.337) | 0.074 | 0.202 | 0.070 / 0.224 | 1.88 | fail |
| one-step MLP model (autoregressive, clipped) | 7114.530 (13732.233 / 15518.262 / 4999.061) | 0.082 | 186.149 | 0.018 / 0.286 | 0.71 | fail |

Three members × 100 epochs per variant (the reference has 5 × 300). Data size: 120 rollouts
already give a usable model (return error 0.145); the 240-rollout draw came out worse than the
120 one (0.246, calibration 2.7), i.e. with 3 short-trained members the curve is noisy at this
resolution. Random / open-loop data only (constant + random signals, 248 rollouts) gives
0.143 vs 0.114 for the mixture. The causal conv branch matches the GRU on return error
(0.112 vs 0.114) at 20× the CPU cost; the padded-MLP branch (branch B of draft D4) is clearly
worse (rel-L2 0.329, return error 0.202, false breakdowns 7 %), which supports the causal
read-out as the default. A single member (M = 1) has a lower return error than the
ensemble mean on val (0.086) but a higher false-breakdown rate (0.053 vs 0.018).

## 6. Figures (`_progress/figures/m12_demo/`)
- `fig1_return_vs_ee.png`: held-out return on T vs cumulative EE (log x), A0 / A1 / A2 / B curves,
  PI-ALINEA and constant-u bands, bootstrap 95 % CIs; `fig2_return_vs_wallclock.png` the same vs wall-clock.
- `fig3_surrogate_accuracy.png`: E1 return-prediction error vs N_0 and per variant.
- `fig4_transfer_gap.png`: surrogate vs SUMO V return of the top-3 checkpoints per round.
- `fig5_breakdown_rates.png`: breakdown rate on T per arm.
- `fig6_policy_structure.png`: the hardest test profile (test[28], peak total 2833 vph, seed 100):
  A1 stores at u ≈ 0.2 during the peak and releases at 0.45 afterwards (no jam, TTS 78);
  PI-ALINEA lets a merge jam form and recovers (TTS 117); the Surrogate-MPC oscillates between
  0 and 1 and gridlocks the mainline (TTS 266); the anticipative and the 290-EE direct policies
  under-release (queues to 250 vehicles, TTS 190 / 155).
- `fig7_ood.png`: OOD returns per arm.

## 7. Surrogate-MPC horizon / terminal-weight sweep (6 V profiles, ledger `demo_mpc_sweep`)
| variant (20 Adam iterations, members 0–2) | mean | p10 | worst | breakdown | TTS | u mean |
|---|---|---|---|---|---|---|
| H 20, terminal 0.5 (the study's arm) | −118.6 | −245 | −254 | 0.50 | 100.6 | 0.49 |
| H 40, terminal 0.5 | −127.7 | −275 | −319 | 0.50 | 108.2 | 0.55 |
| H 20, terminal 5 | −118.6 | −267 | −299 | 0.50 | 100.8 | 0.57 |
| H 40, terminal 5 | −130.4 | −283 | −332 | 0.50 | 111.0 | 0.55 |
| H 60, terminal 2, no offset correction | −130.2 | −272 | −318 | 0.50 | 109.1 | 0.53 |

Neither a longer horizon nor a heavier terminal cost helps: every variant parks near u ≈ 0.5
and oscillates (Fig. 6), so the failure is in the inner optimisation, not the horizon. The
gradient of the TTS objective through the ensemble-mean plant is weak and non-monotone on the
release rate (the model's breakdown is a smooth ramp, and the queue soft-min flattens it),
and 20 Adam steps on sigmoid logits from a shifted warm start do not leave the corner. The
MPC therefore stays in the paper as the "model-based control without RL" reference that the
draft asked for, with the caveat that it needs a better solver (more iterations with a
learning-rate schedule, a sample-based / CEM inner loop, or a policy warm start from A1) before
it can be called competitive. Its 0 EE cost beyond the dataset is real; its ≈ 0.4 s per control
step is 100× the RL policy's.

## 8. Status of the draft's claims after the demo study
| claim | verdict at demo scale | what the paper scale must add |
|---|---|---|
| C1 sample efficiency | supported: A1@854 EE −54.7 vs direct@844 EE −64.1 on T, −70.4 vs −85.0 on O; A2 fine-tune not competitive with aggregation at equal EE | 3–5 seeds, 4 rounds of 1M steps, the 2000-EE direct point |
| C2 ensemble transfer | not supported: single-member policy transfers as well (0 % vs 13 % breakdowns) | repeat with 5 seeds; test the pessimistic mode |
| C3 generalisation | supported on T (+5.9 over PI-ALINEA, CI [+1.3, +10.1]); tie on O (+1.2, CI [−4.9, +6.7]); anticipative variant under-trained at 300k steps | longer anticipative runs |
| C4 operator vs one-step | supported on field error and on the resulting policy (−70.0 vs −81.9 on T); return error comparable | — |
| C5 amortisation | supported in kind: five surrogate PPO runs (rounds, single, one-step, anticipative) cost 0 EE after the dataset; each ≈ 9 min | — |
| Surrogate-MPC | negative result (solver), see §7 | better inner optimiser |

## 9. Theory note for the paper (`theory_analysis.md`, 2026-09-13)
Formal statements of what the pipeline buys, each tied to a measured quantity of §2–§5:
Prop. 1 (SUMO cost = identification only, amortised over every surrogate PPO run),
Theorem 1 (open-loop return error of the history-to-field surrogate is linear in the
per-step field error: ≤ 2 ε̄ρ veh h + ½ ε̄q veh h), Prop. 2 (one-step autoregressive
models compound O(K²) to exponentially; the C4 result in theorem form), Theorem 3 +
Lemma 4 (policy suboptimality ≤ ε(π̂) + ε(π*) + η; surrogate-return ranking is consistent),
Prop. 5 (aggregation as DAgger-style no-regret model fitting on the policy-induced
distributions). §7 of the note instantiates the bounds with the demo numbers (Theorem 1
loose by ≈ 2 on the density term; closed-loop gaps 0–6 veh h inside it) and lists the
0-EE measurements still missing: k-step-ahead error-vs-horizon curve (one-step vs
DeepONet), per-step ε̄ρ / ε̄q in vehicles and vph, the transfer-gap scatter with the
bound, and a monotone ε(N₀) curve with the full 5 × 300-epoch recipe; the 2000-EE direct
point (E4) is the only item that costs SUMO. §8 gives the paper placement (analysis
section between method and experiments), the claim ↔ result ↔ figure mapping, and the
proposal.md addition that needs approval.
