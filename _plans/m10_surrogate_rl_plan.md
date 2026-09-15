# M10 plan — Surrogate RL and the aggregation loop (draft step 3, E3)

Source: `draft_pipeline.md` §6.2, §7, §12 step 3. Status: `_progress/m10_surrogate_rl_progress.md`.

## Deliverables
| Item | Location |
|---|---|
| Batched surrogate env | `src/rl/surrogate_vec_env.py::SurrogateVecEnv` (SB3 VecEnv; modes sample / mean / pessimistic; profile family or fixed set; plant abstraction for DeepONet and one-step models) |
| Reward | `src/rl/reward.py` (`form: tts` primary, `three_term` ablation; `q_ref_mode: offered`; terminal cost; conservation backlog) |
| PPO trainer | `src/rl/train_ppo.py`: `--config ppo_common.yaml --overlay env_*.yaml`, `--set k=v`, `--init-policy`, profile-cycling evaluation on V for both envs, a checkpoint per evaluation pass, ledger lines for SUMO episodes |
| Configs | `configs/rl/ppo_common.yaml`, `env_surrogate.yaml`, `env_sumo.yaml`, `env_sumo_finetune.yaml`, `reward_three_term.yaml` |
| Aggregation loop | `scripts/run_aggregation_loop.py` (per-study forked store, top-k SUMO rollouts on V, fine-tune, spread before/after, stop rule, A0 / A1 outputs) |
| Selection | `scripts/select_surrogate_arm.py`, `scripts/select_checkpoint_profiles.py` |
| Ledger | `src/utils/ledger.py` (`runs/ledger/<study>.jsonl`) |
| Tests | `tests/test_plant_surrogate.py::test_vec_env_contract` (obs / info parity keys, warm-up mask, modes, fixed-set rollouts) |

## PPO configuration (one file, D8)
3×512 tanh MLPs, symmetric action box, `action_init_u` 0.3, `log_std_init` −2, lr 1e-4,
clip 0.2, target_kl 0.02, n_steps 480 (× 16 envs on the surrogate), batch 120,
5 epochs, γ 0.99, λ 0.95. Evaluation: the 18 V profiles, deterministic, every 24k steps
(surrogate, in `mean` ensemble mode) / every 2.4k–24k steps (SUMO, each pass costs 18 EE
and is logged as `eval_val`).

## Round j of the loop
1. PPO S_j steps on ensemble_{j−1} (warm start from the previous selected checkpoint);
2. rank checkpoints by surrogate V return, top k = 3;
3. roll them in SUMO on V (54 EE, purpose `aggregation`) → transfer measurement + data;
4. append to the study store (round j), fine-tune every member;
5. log surrogate vs SUMO return per checkpoint, ensemble spread before / after, cumulative EE.
A0 = round-1 top checkpoint by surrogate return; A1(j) = best SUMO V return of round j.
Stop when the best SUMO V return improves by < 2 or after R rounds.

## Exit criteria
A1 curve on V (this machine: 1 seed at demo scale; paper: 5 seeds, 1M steps/round, R = 4),
ledger complete, transfer-gap log per round.
