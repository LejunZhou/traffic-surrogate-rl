# M9 plan — Plant-model DeepONet v2 (draft step 2, E1, E2)

Source: `draft_pipeline.md` §5, §6.1, §7.2, §12 step 2. Status: `_progress/m9_plant_model_v2_progress.md`.

## Goal
Learn the plant operator G: (d(·), q_r(·)) ↦ (ρ(x,t), q(x,t)) from a budgeted round-0
set of closed-loop SUMO rollouts, as a bootstrap ensemble with a causal branch, and
show it is accurate enough for RL (round-0 gate) and reward-consistent (E2 parity).

## Deliverables
| Item | Location |
|---|---|
| Round-0 mixture dataset | `scripts/generate_round0_dataset.py`, `configs/experiments/round0_mixture.yaml`, store `data/plant_v2/round0` |
| DeepONet v2 | `surrogate/deeponet.py`: `CausalConvBranch`, `GRUBranch`, `PaddedMLPBranch`, `PlantTrunk`, `PlantDeepONet`, `PlantNormalisation`, `DeepONetEnsemble` |
| Dataset | `surrogate/datasets.py::PlantRolloutDataset` (causal / padded views, band weights, exit-flow labels, bootstrap repeats) |
| Trainer | `surrogate/train_plant.py` (`--member`, `--bootstrap-seed`, `--resume --finetune --new-rounds`, `--max-train-files`, `--train-types`) |
| Ensemble launcher | `scripts/train_ensemble.py` → `member_i/best.pt` + `manifest.json` |
| Evaluation | `surrogate/eval_plant.py`, `scripts/eval_surrogate_regimes.py` (per-regime rel-L2, return-prediction error, breakdown onset / false rate, calibration) |
| E2 parity | `scripts/reward_parity_check.py` |
| E1 study | `scripts/run_e1_surrogate_study.py` |
| One-step baseline (C4) | `surrogate/onestep.py`, `configs/surrogate/onestep_v1.yaml` |
| Configs | `configs/surrogate/plant_v2.yaml` (GRU), `plant_v2_conv.yaml`, `plant_v2_mlp_padded.yaml` |

## Architecture as built (vs draft §5.3)
- Branch input (2, K) = [d/2500, q_r/1600]; branch read-out b_k ∈ R^256 at every k.
- Default branch = 2-layer GRU (128 hidden). The draft's dilated causal conv stack was
  implemented (`CausalConvBranch`, residual, per-position LayerNorm — a GroupNorm leaked
  across time and was caught by the causality test) but on CPU it costs 120 ms per
  16-sequence forward vs 5 ms for the GRU with the same latent width, and the GRU steps
  incrementally (0.06 ms), which the MPC exploits. The conv branch is the E1 ablation.
- Trunk MLP 2 → 512 → 512 → 512 → 2p (GELU; optional Fourier features off), split into
  τ_ρ and τ_q; ρ̂ = ⟨b_k, τ_ρ⟩/√p + b_ρ (z-scored), q̂ = ⟨b_k, τ_q⟩/√p + b_q (/2500).
- Loss: band-weighted MSE on 512 density queries per rollout per epoch (+2 in the
  shockwave band ρ > 40 upstream of 1400 m) + 1.0 × MSE on the exit flow at all K steps.
- AdamW 1e-3, cosine with 5 warm-up epochs, wd 1e-6, batch 16, clip 1.0, 300 epochs
  (3.5–6.5 s/epoch on 486 rollouts), selection every 5 epochs on validation
  rel-L2(ρ) + rel-L2(q_exit) in physical units.
- Ensemble: 5 members, seeds 0–4, bootstrap resample of the round-0 train split; the
  manifest carries geometry + normalisation so `SurrogateVecEnv` needs no SUMO config.
- Fine-tune: resume, original bootstrap list + every rollout of the new rounds, 20
  epochs at 3e-4, best.pt re-selected on the round-0 validation split.

## Gates
- Round-0 gate (§5.5): return-prediction error ≤ 10 %, false breakdown rate ≤ 10 %,
  calibration slope ∈ [0.5, 2] on the validation split.
- E2: every reward term's episode-sum range from the surrogate within 20 % of SUMO's.
