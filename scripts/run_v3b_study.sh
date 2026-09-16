#!/bin/sh
# Scenario-v3b study driver (M14): E0 -> round-0 dataset -> plant ensemble -> gate
# -> aggregation loop (A0/A1) || direct SUMO PPO (B) -> ALINEA / constant baselines
# -> manifest -> final evaluation on T and O -> figures. Every stage is idempotent.
# Lean version of run_study.sh (no MPC / A2 / single-surrogate / one-step / E1 arms).
#
#   sh scripts/run_v3b_study.sh                       # demo scale, seed 0, 8 workers (~5 h on 10 cores)
#   SMOKE=1 sh scripts/run_v3b_study.sh               # 30-min end-to-end smoke test on isolated paths
#   SEEDS="0 1 2" STEPS_PER_ROUND=1000000 ROUNDS=4 DIRECT_EE="200 700 2000" sh scripts/run_v3b_study.sh   # paper scale
#
# The scenario is selected by SCENARIO (default v3b): configs/rl/env_$SCENARIO.yaml
# is applied to every env (surrogate and SUMO) through the SCENARIO_OVERLAY
# environment variable (rl.profile_eval.scenario_overlays), the frozen profile
# sets come from PROFILE_SETS_DIR, the round-0 config is configs/experiments/round0_$SCENARIO.yaml.
set -e
cd "$(dirname "$0")/.."
# project venv if present (macOS/Linux: bin, Windows Git Bash: Scripts); an already
# activated conda env (CLAUDE.md) works too — the script only needs python, sumo and
# netconvert on PATH
[ -d .venv-traffic-rl/bin ] && export PATH="$PWD/.venv-traffic-rl/bin:$PATH"
[ -d .venv-traffic-rl/Scripts ] && export PATH="$PWD/.venv-traffic-rl/Scripts:$PATH"
[ -x .venv-traffic-rl/python.exe ] && export PATH="$PWD/.venv-traffic-rl:$PATH"   # Windows conda env: python.exe sits at the env root
export PYTHONPATH=src
command -v sumo >/dev/null 2>&1 || { echo "sumo not on PATH (activate the project env or install eclipse-sumo)"; exit 1; }
SCENARIO=${SCENARIO:-v3b}
MILESTONE=${MILESTONE:-m14}          # prefix of the E0 report, ALINEA report, ledgers and figure dir (M15: MILESTONE=m15 SCENARIO=v4)
export SCENARIO_OVERLAY=${SCENARIO_OVERLAY:-configs/rl/env_$SCENARIO.yaml}
export PROFILE_SETS_DIR=${PROFILE_SETS_DIR:-configs/profiles/v2}
ROUND0_CFG=${ROUND0_CFG:-configs/experiments/round0_$SCENARIO.yaml}
PLANT_CFG=${PLANT_CFG:-configs/surrogate/plant_v2.yaml}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  STUDY=${STUDY:-${SCENARIO}_smoke}
  STORE=${STORE:-data/plant_$SCENARIO/smoke_round0}
  ENS=${ENS:-runs/surrogate/plant_${SCENARIO}_smoke}
  N_ROUND0=${N_ROUND0:-40}; MEMBERS=${MEMBERS:-2}; EPOCHS=${EPOCHS:-2}
  ROUNDS=${ROUNDS:-1}; STEPS_PER_ROUND=${STEPS_PER_ROUND:-4800}; EVAL_FREQ=${EVAL_FREQ:-240}; FT_EPOCHS=${FT_EPOCHS:-1}
  DIRECT_EE=${DIRECT_EE:-"4"}; ALINEA_ARGS=${ALINEA_ARGS:-"--stage1-profiles 1 4 --dets 12 --rhos 30 --kis 20 --top 1"}
  FINAL_SETS=${FINAL_SETS:-"val"}; SKIP_E0=${SKIP_E0:-1}
else
  STUDY=${STUDY:-$SCENARIO}
  STORE=${STORE:-data/plant_$SCENARIO/round0}
  ENS=${ENS:-runs/surrogate/plant_${SCENARIO}_r0}
  N_ROUND0=${N_ROUND0:-0}; MEMBERS=${MEMBERS:-5}; EPOCHS=${EPOCHS:-0}
  ROUNDS=${ROUNDS:-3}; STEPS_PER_ROUND=${STEPS_PER_ROUND:-300000}; EVAL_FREQ=${EVAL_FREQ:-0}; FT_EPOCHS=${FT_EPOCHS:-20}
  DIRECT_EE=${DIRECT_EE:-"200 700"}; ALINEA_ARGS=${ALINEA_ARGS:-""}
  FINAL_SETS=${FINAL_SETS:-"test ood"}; SKIP_E0=${SKIP_E0:-0}
fi
SEEDS=${SEEDS:-"0"}
WORKERS=${WORKERS:-8}
STOP_DELTA=${STOP_DELTA:-2.0}
STOP_PATIENCE=${STOP_PATIENCE:-2}
OUT=runs/study/$STUDY
E0_REPORT=_progress/${MILESTONE}_e0_${SCENARIO}_characterisation.json
mkdir -p $OUT runs/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# study-specific overlay: density normalisation from this study's round-0 store
# (the scenario overlay names the default store; smoke runs use their own)
printf 'env:\n  density_stats_from: %s/metadata.json\n' "$STORE" > $OUT/env_study.yaml
export SCENARIO_OVERLAY="$SCENARIO_OVERLAY:$OUT/env_study.yaml"
OVERLAY_ARGS=$(for ov in $(echo $SCENARIO_OVERLAY | tr ':' ' '); do printf -- "--overlay %s " $ov; done)
log "study $STUDY: scenario $SCENARIO (overlay $SCENARIO_OVERLAY, profiles $PROFILE_SETS_DIR), store $STORE, ensemble $ENS"

# ---------------------------------------------------------------- E0 (its rollouts seed the round-0 store)
if [ "$SKIP_E0" != "1" ] && [ ! -f $E0_REPORT ]; then
  log "E0 characterisation"
  python scripts/run_scenario_characterisation.py --config $ROUND0_CFG --out $E0_REPORT --study ${MILESTONE}_e0_$SCENARIO --workers $WORKERS > runs/logs/${STUDY}_e0.log 2>&1
fi

# ---------------------------------------------------------------- round-0 dataset
if [ ! -f $STORE/split_index.json ]; then
  log "round-0 dataset"
  NARG=""; [ "$N_ROUND0" != "0" ] && NARG="--n $N_ROUND0"
  python scripts/generate_round0_dataset.py --config $ROUND0_CFG --store-dir $STORE --network-dir $STORE/../network_${STUDY} \
      --workers $WORKERS --study ${STUDY}_r0 $NARG > runs/logs/${STUDY}_r0.log 2>&1
fi

# ---------------------------------------------------------------- plant ensemble + gate
if [ ! -f $ENS/manifest.json ]; then
  log "plant ensemble ($MEMBERS members)"
  EARG=""; [ "$EPOCHS" != "0" ] && EARG="--epochs $EPOCHS"
  python scripts/train_ensemble.py --config $PLANT_CFG --out-dir $ENS --store-dir $STORE --members $MEMBERS --parallel $MEMBERS $EARG > runs/logs/${STUDY}_ens.log 2>&1
fi
[ -f $ENS/eval_val.json ] || python scripts/eval_surrogate_regimes.py --ensemble $ENS --store $STORE --split val > runs/logs/${STUDY}_gate_val.log 2>&1
[ -f $ENS/eval_test.json ] || python scripts/eval_surrogate_regimes.py --ensemble $ENS --store $STORE --split test > runs/logs/${STUDY}_gate_test.log 2>&1
log "round-0 gate on val: $(python -c "import json;print(json.load(open('$ENS/eval_val.json'))['gate'])")"

# ---------------------------------------------------------------- aggregation loop (A0, A1) per seed  ||  direct SUMO PPO (B)
SETARGS=""; [ "$EVAL_FREQ" != "0" ] && SETARGS="--set training.eval_freq=$EVAL_FREQ"
for s in $SEEDS; do
  [ -f runs/aggregation/${STUDY}_s$s/study.json ] || \
    python scripts/run_aggregation_loop.py --study ${STUDY}_s$s --seed $s --rounds $ROUNDS --steps-per-round $STEPS_PER_ROUND \
      --ensemble $ENS --store $STORE --workers $WORKERS --stop-delta $STOP_DELTA --stop-patience $STOP_PATIENCE \
      --finetune-epochs $FT_EPOCHS $SETARGS > runs/logs/${STUDY}_agg_s$s.log 2>&1 &
done
for ee in $DIRECT_EE; do
  for s in $SEEDS; do
    run=$OUT/direct_ppo_${ee}ee_s$s
    steps=$((ee * 120))
    if [ "$EVAL_FREQ" != "0" ]; then evalf=$EVAL_FREQ; else evalf=$(( ee <= 200 ? 4800 : (ee <= 700 ? 9600 : 24000) )); fi
    [ -f $run/final_model.zip ] || python -m rl.train_ppo --config configs/rl/ppo_common.yaml --overlay configs/rl/env_sumo.yaml $OVERLAY_ARGS \
        --seed $s --total-timesteps $steps --set training.eval_freq=$evalf --set training.ledger_study=${STUDY}_direct_${ee}_s$s \
        --set output.run_dir=$run --set env.network_dir=data/raw/rl_network_${STUDY}_direct_${ee}_s$s > runs/logs/${STUDY}_direct_${ee}_s$s.log 2>&1 &
  done
done

# ---------------------------------------------------------------- baselines: ALINEA tuning + constants (SUMO workers)
ALINEA_JSON=_progress/${MILESTONE}_alinea_tuning_${STUDY}.json
[ -f $ALINEA_JSON ] || python scripts/tune_alinea_profiles.py --workers $WORKERS --study ${STUDY}_alinea --out $ALINEA_JSON $ALINEA_ARGS > runs/logs/${STUDY}_alinea.log 2>&1
wait
BEST_ALINEA=$(python -c "import json;print(json.load(open('$ALINEA_JSON'))['best_alinea'])")
BEST_CONST=$(python -c "import json;print(json.load(open('$ALINEA_JSON'))['best_constant'])")
log "aggregation and direct PPO finished; ALINEA: $BEST_ALINEA  constant: $BEST_CONST"

# ---------------------------------------------------------------- checkpoint selection of the direct arm on V
for ee in $DIRECT_EE; do for s in $SEEDS; do
  run=$OUT/direct_ppo_${ee}ee_s$s
  [ -f $run/best_model_selected.zip ] || python scripts/select_checkpoint_profiles.py --run $run > runs/logs/${STUDY}_select_direct_${ee}_s$s.log 2>&1
done; done

# ---------------------------------------------------------------- manifest, final evaluation, figures
python scripts/build_arms_manifest.py --study $STUDY --seeds $SEEDS --direct-ee $DIRECT_EE --alinea "$BEST_ALINEA" --constant "$BEST_CONST" \
    --ensemble $ENS --store $STORE --r0-studies ${STUDY}_r0 ${MILESTONE}_e0_$SCENARIO --no-mpc --out $OUT/arms.json
python scripts/run_final_evaluation.py --arms $OUT/arms.json --workers $WORKERS --study ${STUDY}_final --sets $FINAL_SETS > runs/logs/${STUDY}_final_eval.log 2>&1
python scripts/plot_sample_efficiency.py --arms $OUT/arms.json --rounds $(ls runs/aggregation/${STUDY}_s*/rounds.json) --e1 "" --out _progress/figures/${MILESTONE}_$STUDY
log "study $STUDY complete: $OUT"
