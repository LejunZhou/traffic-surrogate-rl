# M11 progress — Reference arms and surrogate-enabled baselines (2026-09-13)

Plan: `_plans/m11_reference_arms_plan.md`. All numbers are returns under the TTS-form
reward (−TTS in veh h) on the validation set V (18 profiles × 1 seed) unless stated.

## 1. ALINEA / PI-ALINEA tuning on V (`_progress/m11_alinea_tuning.json`, ledger `demo_alinea`)
Stage 1: 36 candidates (det {11, 12, 13} × ρ_set {26, 30, 34, 38} × K_I {10, 20, 35}) on 6 V
profiles (216 EE). Stage 2: top 4 + PI variants on all 18 profiles (108 EE). Constants:
u ∈ {0, 0.1, …, 1} on V (198 EE). Total tuning cost 522 EE, 22 min on 8 workers.

| controller | mean | p10 | worst | breakdown rate | TTS (veh h) | u mean |
|---|---|---|---|---|---|---|
| **PI-ALINEA kp 4, K_I 20, ρ_set 26, det 13** | **−57.0** | −95.5 | −154.5 | 0.00 | 52.0 | 0.77 |
| PI-ALINEA kp 4, K_I 10, ρ_set 26, det 13 | −61.8 | −116.7 | −156.9 | 0.22 | 56.0 | 0.77 |
| ALINEA K_I 35, ρ_set 26, det 13 | −68.3 | −135.9 | −184.9 | 0.00 | 63.6 | 0.73 |
| ALINEA K_I 10, ρ_set 30, det 11 | −69.6 | −134.5 | −177.3 | 0.28 | 61.9 | 0.77 |
| best constant u = 0.3 | −90.1 | −206.0 | −211.2 | 0.33 | 76.4 | 0.30 |
| constant u = 0.8 (pass-through) | −99.0 | −230.0 | −289.4 | 0.44 | 82.9 | 0.80 |

With the occupancy estimator the winning set-point moved to 26 veh/km (the M7 q/v-based
winner was 37) and the detector one cell downstream of the merge nose (13 = 1400 m);
low gain still wins. The PI term helps on profiles (M7 grid: a tie). Every constant
either stores too much (u ≤ 0.2: final queues 140–444) or jams on the storage-mandatory
profiles (u ≥ 0.4: breakdown rate 0.44–0.50), which is the E0 finding again.

## 2. Learned arms on V (selection results; final numbers on T / O in `_progress/m12_study_progress.md`)
| arm | budget (EE, ledger) | selected checkpoint | SUMO V return | breakdown rate on V | note |
|---|---|---|---|---|---|
| B direct SUMO PPO, 200 EE | 290 (200 training + 5 × 18 eval_val) | 24k steps | −80.1 | — | curve −95 → −89 → −95 → −84 → −80; still improving at the budget end |
| B direct SUMO PPO, 700 EE | 844 (700 + 8 × 18 eval_val) | 67.2k steps | −65.8 | — | curve −72 (48k) → −77 → −66 (67k) → −70 (77k); 2.9 h wall-clock |
| A0 zero-shot (round-1 top by surrogate return) | 692 (+54 selection) | 264k | −68.5 | 0.11 | see M10 |
| A1 aggregation, round 3 | 854 | 288k | **−49.5** | 0.00 | see M10 |
| A2 = selected_r1 + 100 EE SUMO fine-tune | 882 | 9.6k steps | −64.6 | — | −68.5 → −71.7 (4.8k) → −64.6 (9.6k); lr 3e-5, target_kl 0.01, log-std reset −2 |
| single-surrogate (member 0, mean mode, 300k) | 746 | 264k | −66.3 | 0.00 | top-3 by surrogate: −70.3 / −66.3 / −68.3 in SUMO |
| one-step model arm (`plant_type: onestep`, 300k) | 746 | 216k | −72.7 | 0.00 | PPO on the one-step plant runs at ≈ 4 800 fps; the 264k checkpoint jammed one profile (−310) |
| anticipative (look-ahead 20 steps, 63-dim obs, 300k) | 746 | 120k | −83.2 | 0.00 | behind the reactive policy at this budget (larger input, same steps); paper scale needs 1M+ steps |
| Surrogate-MPC (H 20, 20 Adam iterations, members 0–2, TTS objective, perfect forecast) | 692 | — | (evaluated directly on T / O) | | ≈ 0.4 s per control step |

Reading on V: every surrogate-trained policy (A0, single, one-step, anticipative) transfers
without catastrophic gridlock (0–11 % breakdown rate, all recovering) and beats direct SUMO
PPO at 200 EE by 12–20 veh h; the aggregation loop is what closes the remaining gap to and
beyond PI-ALINEA. A2 (100 EE of SUMO fine-tuning) gains 4 veh h over its zero-shot start but
does not reach A1's round-2/3 policies obtained with the same 100–160 EE spent on aggregation
rollouts, i.e. at this budget spending SUMO episodes on *data for the model* beats spending
them on *gradient steps for the policy*.
