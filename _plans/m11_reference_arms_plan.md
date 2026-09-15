# M11 plan — Reference arms and surrogate-enabled baselines (E4, E5, E6)

Source: `draft_pipeline.md` §8, §9, §11. Status: `_progress/m11_reference_arms_progress.md`.

| Arm | How | Budget (EE) | Script |
|---|---|---|---|
| B direct SUMO PPO | run-7 recipe on the profile family, `env_sumo.yaml`, budgets 200 / 700 / 2000 EE (+ eval_val passes), post-hoc selection on V | 200–2000 | `rl.train_ppo`, `select_checkpoint_profiles.py` |
| A2 pre-train + fine-tune | `env_sumo_finetune.yaml` + `--init-policy selected_r1.zip`: value net kept, lr 3e-5, target_kl 0.01, log-std reset −2, 100 / 200 EE | 480 + 54 + 100/200 | same |
| ALINEA / PI-ALINEA | two-stage tuning on V (36 candidates × 6 profiles, then top 4 + PI on 18) | ≈ 300 | `tune_alinea_profiles.py` |
| constant u | 11 constants × 18 V profiles | 198 | same |
| Surrogate-MPC | gradient MPC through the ensemble-mean DeepONet, H = 20, 30 Adam iterations, soft-min queue, offset correction, TTS objective, perfect demand forecast | 0 beyond the dataset | `rl/surrogate_mpc.py` (`mpc:<dir>,iters=,members=`) |
| single-surrogate control | A0 recipe with one deterministic member (`env.members=[0]`, mode mean) | 480 + 54 | `run_study.sh` |
| one-step model arm | PPO on `plant_type: onestep` (same env otherwise) | 480 + 54 | same |
| anticipative (E7) | `observation.lookahead_steps: 20` on the surrogate | 480 + 54 | same |

All arms are selected on V only; T and O are touched once in M12.
