# M14 rerun with a 60 km/h ramp, on Google Colab (2026-09-22)

## Why
The paper's M14 study (v3b, 120 km/h ramp) is rerun from scratch with a 60 km/h ramp. The rerun also
fixes the paper-audit gaps (M14 progress §17–§18): Q_{k+1} queue timing in both rewards (done), a clean
direct SUMO-PPO arm at 1000 episodes, Surrogate-MPC in the evaluation, and a script that builds
Tables I and II. The whole study runs on Colab: the local Windows machine lost two overnight runs to restarts.

## Decisions (user, 2026-09-22)
- Ramp 60 km/h (`m14/configs/scenario.yaml`); m14/ edited in place.
- Reward charges Q_{k+1} in SUMO and the surrogate.
- Direct SUMO-PPO: one fresh run at 1000 episodes (seed 0); validation every 9600 steps.
- Surrogate-MPC included (option a); its negative result needs an explanation later (open item).
- Table script with **summed compute time** (ledger episode walls + ensemble member training + PPO process time).
- Seed 0 only now; seeds 1–2 later.
- ALINEA grid widened so the tuned baseline is not limited by its search (user, 2026-09-23): det 12–15,
  ρ̂ 20–34, ki 10/20/35, kp 2/4/8 (540 episodes); a selected value on the grid edge triggers a wider retune.

## Changes
1. `rl/train_ppo.py`: `training.resume` — continue a direct SUMO run from its latest checkpoint across
   Colab sessions (optimizer state kept; evaluation history, logs and ledger kept consistent; episodes after
   the checkpoint moved to `ledger_lost.jsonl` in the run dir; profile/SUMO-seed streams continue, not replayed).
2. `run.py`: `sumo-ppo` resumes instead of refusing, default budget 1000, eval every 9600 steps for
   budgets > 200; `baselines` grid options; `evaluate` adds PI-ALINEA and ALINEA as separate arms and
   Surrogate-MPC on the final aggregation ensemble; new `e0` and `tables` stages; `pipeline --parallel`
   runs [DeepONet → surrogate-PPO] ‖ direct PPO ‖ baselines after the data stage.
3. `tune_alinea_profiles.py`: `--kps`, best pure ALINEA and best PI-ALINEA recorded separately.
4. `build_arms_manifest.py`: ALINEA / PI-ALINEA arms with their own costs; MPC cost = the data and model
   training behind its ensemble (no PPO time).
5. New `scripts/build_paper_tables.py` (Table I, Table II, TTS reductions with paired bootstrap CIs) and
   `scripts/compare_e0_capacity.py` (capacity at 60 vs 120 km/h, reference report copied into m14/).
6. `colab/m14_ramp60.ipynb`: Drive working copy, SUMO wheel, capacity gate, background pipeline, status, tables.

7. (2026-09-23, after seed 0) Incremental GRU branch in the surrogate environment (same outputs, ≈ 5–8× faster
   surrogate PPO) and a CPU thread cap for PPO processes (`run.py --torch-threads`), so seeds 1–2 can run in parallel.
8. (2026-09-24, after seeds 0–2) Direct PPO at a compute budget at least equal to Surrogate-PPO's: `run.py
   extend-direct` continues each seed's 1000-episode run to 1200 episodes (≈ 1470 SUMO episodes, ≈ 6.6 h summed
   compute vs 5.9–6.3 h for Surrogate-PPO with the fast code), tables with both budgets; `colab/m14_direct_extend.ipynb`.
   Why: with the fast code Surrogate-PPO costs 0.5–0.9 h more compute than direct PPO at 1000 episodes, so a
   reviewer can ask whether direct PPO simply needed that time. Same-seed continuation (constant learning rate,
   optimizer state kept) is equivalent to having trained longer and costs ≈ 1 h instead of ≈ 5.5 h for new runs.

## Verification
Unit tests (resume bookkeeping, table maths); `run.py smoke` extended with an interrupted-and-resumed direct
PPO run and MPC; notebook cells checked for syntax (Colab itself cannot be exercised locally).

## Open
- ALINEA grid choice; MPC explanation; seeds 1–2; paper text edits (Eq. 4, Theorem 1, 60 km/h, cost column).
