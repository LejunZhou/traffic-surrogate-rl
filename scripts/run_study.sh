#!/bin/sh
# End-to-end M8-M12 study driver (draft_pipeline.md §11-§12).
#
# Runs every experiment after the round-0 dataset and ensemble exist, at a
# budget set by the variables below (defaults = the CPU "demo" scale that
# completes in ~3-4 h on a 10-core laptop; the paper-scale values are given in
# the comments). Every stage is idempotent: re-running skips finished outputs.
#
#   sh scripts/run_study.sh                      # demo scale
#   STUDY=paper STEPS_PER_ROUND=1000000 ROUNDS=4 DIRECT_EE="200 700 2000" SEEDS="0 1 2" sh scripts/run_study.sh
set -e
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-traffic-rl/bin:$PATH"
export PYTHONPATH=src
STUDY=${STUDY:-demo}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-300000}      # paper: 1000000
ROUNDS=${ROUNDS:-3}                              # paper: 4
DIRECT_EE=${DIRECT_EE:-"200 700"}                # paper: 200 700 2000
SEEDS=${SEEDS:-"0"}                              # paper: 0 1 2 (surrogate arm: 0-4)
FT_EE=${FT_EE:-100}                              # A2 fine-tune budget (100 or 200)
WORKERS=${WORKERS:-8}
E1_MEMBERS=${E1_MEMBERS:-3}
E1_EPOCHS=${E1_EPOCHS:-150}
MPC_SPEC=${MPC_SPEC:-"mpc:runs/surrogate/plant_v2_round0,iters=20,members=0-2"}   # paper: iters=30, all members
ENS=runs/surrogate/plant_v2_round0
STORE=data/plant_v2/round0
OUT=runs/study/$STUDY
mkdir -p $OUT runs/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------- M9: E2 parity
[ -f _progress/m9_e2_parity.json ] || python scripts/reward_parity_check.py --ensemble $ENS --store $STORE

# ---------------------------------------------------------------- M10: aggregation (A0, A1) per seed
for s in $SEEDS; do
  [ -f runs/aggregation/${STUDY}_s$s/study.json ] || \
    python scripts/run_aggregation_loop.py --study ${STUDY}_s$s --seed $s --rounds $ROUNDS --steps-per-round $STEPS_PER_ROUND \
      --ensemble $ENS --store $STORE --workers $WORKERS > runs/logs/${STUDY}_agg_s$s.log 2>&1 &
  AGG_PID=$!
done

# ---------------------------------------------------------------- M11: baselines (ALINEA tuning, constants) - SUMO workers
[ -f _progress/m11_alinea_tuning.json ] || python scripts/tune_alinea_profiles.py --workers $WORKERS --study ${STUDY}_alinea > runs/logs/${STUDY}_alinea.log 2>&1
BEST_ALINEA=$(python -c "import json;print(json.load(open('_progress/m11_alinea_tuning.json'))['best_alinea'])")
BEST_CONST=$(python -c "import json;print(json.load(open('_progress/m11_alinea_tuning.json'))['best_constant'])")
log "ALINEA: $BEST_ALINEA  constant: $BEST_CONST"

# ---------------------------------------------------------------- M11: direct SUMO PPO (arm B) per budget and seed
for ee in $DIRECT_EE; do
  for s in $SEEDS; do
    run=$OUT/direct_ppo_${ee}ee_s$s
    steps=$((ee * 120))
    evalf=$(( ee <= 200 ? 4800 : (ee <= 700 ? 9600 : 24000) ))
    [ -f $run/final_model.zip ] || python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_sumo.yaml \
        --seed $s --total-timesteps $steps --set training.eval_freq=$evalf --set training.ledger_study=${STUDY}_direct_${ee}_s$s \
        --set output.run_dir=$run --set env.network_dir=data/raw/rl_network_direct_${ee}_s$s > runs/logs/${STUDY}_direct_${ee}_s$s.log 2>&1 &
  done
done

# ---------------------------------------------------------------- M9: E1 one-step model (needed by the one-step arm)
[ -f runs/surrogate/onestep_round0/manifest.json ] || \
  python scripts/run_e1_surrogate_study.py --members $E1_MEMBERS --epochs $E1_EPOCHS --parallel 3 --only onestep > runs/logs/${STUDY}_e1_onestep.log 2>&1
wait
log "aggregation and direct PPO finished"

# ---------------------------------------------------------------- M11: A2 fine-tune, single-surrogate, one-step arms (seed 0)
S0=$(echo $SEEDS | cut -d' ' -f1)
AGG=runs/aggregation/${STUDY}_s$S0
run=$OUT/finetune_a2_${FT_EE}ee
[ -f $run/final_model.zip ] || python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_sumo_finetune.yaml \
    --init-policy $AGG/selected_r1.zip --seed $S0 --total-timesteps $((FT_EE * 120)) --set training.eval_freq=4800 \
    --set training.ledger_study=${STUDY}_finetune --set training.ledger_purpose=finetune --set output.run_dir=$run \
    --set env.ensemble_dir=$ENS > runs/logs/${STUDY}_finetune.log 2>&1 &
run=$OUT/single_surrogate
[ -f $run/final_model.zip ] || python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_surrogate.yaml \
    --ensemble-dir $ENS --seed $S0 --total-timesteps $STEPS_PER_ROUND --set env.members=[0] --set env.ensemble_mode=mean \
    --set output.run_dir=$run > runs/logs/${STUDY}_single.log 2>&1 &
run=$OUT/onestep_arm
[ -f $run/final_model.zip ] || { [ -f runs/surrogate/onestep_round0/manifest.json ] && \
  python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_surrogate.yaml \
    --ensemble-dir runs/surrogate/onestep_round0 --seed $S0 --total-timesteps $STEPS_PER_ROUND --set env.plant_type=onestep \
    --set output.run_dir=$run > runs/logs/${STUDY}_onestep.log 2>&1 & }
run=$OUT/anticipative
[ -f $run/final_model.zip ] || python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_surrogate.yaml \
    --ensemble-dir $ENS --seed $S0 --total-timesteps $STEPS_PER_ROUND --set env.observation.lookahead_steps=20 \
    --set output.run_dir=$run > runs/logs/${STUDY}_anticipative.log 2>&1 &
wait
log "A2, single-surrogate, one-step and anticipative arms finished"

# ---------------------------------------------------------------- selection of the surrogate-only arms on V (SUMO)
python scripts/select_surrogate_arm.py --run $OUT/single_surrogate --ensemble $ENS --study ${STUDY}_single --workers $WORKERS > runs/logs/${STUDY}_select_single.log 2>&1
[ -d $OUT/onestep_arm ] && python scripts/select_surrogate_arm.py --run $OUT/onestep_arm --ensemble $ENS --study ${STUDY}_onestep --workers $WORKERS > runs/logs/${STUDY}_select_onestep.log 2>&1
python scripts/select_surrogate_arm.py --run $OUT/anticipative --ensemble $ENS --study ${STUDY}_anticipative --workers $WORKERS --lookahead 20 > runs/logs/${STUDY}_select_anticipative.log 2>&1
for ee in $DIRECT_EE; do for s in $SEEDS; do
  python scripts/select_checkpoint_profiles.py --run $OUT/direct_ppo_${ee}ee_s$s > runs/logs/${STUDY}_select_direct_${ee}_s$s.log 2>&1
done; done
python scripts/select_checkpoint_profiles.py --run $OUT/finetune_a2_${FT_EE}ee > runs/logs/${STUDY}_select_ft.log 2>&1

# ---------------------------------------------------------------- M12: final evaluation on T and O + figures
python scripts/build_arms_manifest.py --study $STUDY --seeds $SEEDS --direct-ee $DIRECT_EE --ft-ee $FT_EE \
    --alinea "$BEST_ALINEA" --constant "$BEST_CONST" --ensemble $ENS --mpc-spec "$MPC_SPEC" --out $OUT/arms.json
python scripts/run_final_evaluation.py --arms $OUT/arms.json --workers $WORKERS --study ${STUDY}_final > runs/logs/${STUDY}_final_eval.log 2>&1
python scripts/plot_sample_efficiency.py --arms $OUT/arms.json --rounds $(ls runs/aggregation/${STUDY}_s*/rounds.json) --out _progress/figures/m12_$STUDY
log "study $STUDY complete: $OUT"

# ---------------------------------------------------------------- M9: remaining E1 ablations (torch only, slow) + figure 3
python scripts/run_e1_surrogate_study.py --members $E1_MEMBERS --epochs $E1_EPOCHS --parallel 3 --only sizes,source,branch > runs/logs/${STUDY}_e1.log 2>&1
python scripts/plot_sample_efficiency.py --arms $OUT/arms.json --rounds $(ls runs/aggregation/${STUDY}_s*/rounds.json) --out _progress/figures/m12_$STUDY
log "E1 complete"
