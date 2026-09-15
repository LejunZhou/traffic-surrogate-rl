#!/bin/sh
# M13: round-0 budget and mixture study (`_plans/m13_round0_budget_plan.md`).
#
# Per mix: 240-rollout round-0 dataset -> 5-member plant ensemble -> gate on
# val / test -> aggregation loop (stop rule with patience) ; then a manifest
# that re-uses the demo study's reference arms, the final evaluation on T and O
# of every round's policy, and the figures. Every stage is idempotent.
#
#   sh scripts/run_m13.sh                                   # mix1 + mix2 concurrently, 4 workers each
#   MIX=mix2 CONCURRENT=0 WORKERS=8 sh scripts/run_m13.sh   # one mix, all workers
set -e
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-traffic-rl/bin:$PATH"
export PYTHONPATH=src
MIX=${MIX:-"mix1 mix2"}
ROUNDS=${ROUNDS:-11}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-300000}
SEED=${SEED:-0}
STOP_DELTA=${STOP_DELTA:-2.0}
STOP_PATIENCE=${STOP_PATIENCE:-2}
CONCURRENT=${CONCURRENT:-1}
NMIX=$(echo $MIX | wc -w | tr -d ' ')
if [ -z "$WORKERS" ]; then
  if [ "$CONCURRENT" = "1" ] && [ "$NMIX" -gt 1 ]; then WORKERS=4; else WORKERS=8; fi
fi
OUT=runs/study/m13
mkdir -p $OUT runs/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_mix() {
  m=$1
  STORE=data/plant_v2/round0_s240_$m
  ENS=runs/surrogate/plant_v2_r0s240_$m
  AGG=runs/aggregation/m13_${m}_s$SEED
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
  log "[$m] round-0 dataset"
  [ -f $STORE/split_index.json ] || python scripts/generate_round0_dataset.py --config configs/experiments/round0_small_$m.yaml \
      --workers $WORKERS --study m13_r0_$m > runs/logs/m13_r0_$m.log 2>&1
  log "[$m] ensemble"
  [ -f $ENS/manifest.json ] || python scripts/train_ensemble.py --config configs/surrogate/plant_v2.yaml --out-dir $ENS \
      --store-dir $STORE --members 5 --parallel 5 > runs/logs/m13_ens_$m.log 2>&1
  log "[$m] gate"
  [ -f $ENS/eval_val.json ] || python scripts/eval_surrogate_regimes.py --ensemble $ENS --store $STORE --split val > runs/logs/m13_gate_${m}_val.log 2>&1
  [ -f $ENS/eval_test.json ] || python scripts/eval_surrogate_regimes.py --ensemble $ENS --store $STORE --split test > runs/logs/m13_gate_${m}_test.log 2>&1
  log "[$m] aggregation loop"
  [ -f $AGG/study.json ] || python scripts/run_aggregation_loop.py --study m13_${m}_s$SEED --seed $SEED --rounds $ROUNDS \
      --steps-per-round $STEPS_PER_ROUND --ensemble $ENS --store $STORE --workers $WORKERS \
      --stop-delta $STOP_DELTA --stop-patience $STOP_PATIENCE --finetune-epochs 20 > runs/logs/m13_agg_$m.log 2>&1
  log "[$m] done"
}

if [ "$CONCURRENT" = "1" ] && [ "$NMIX" -gt 1 ]; then
  for m in $MIX; do run_mix $m & done
  wait
else
  for m in $MIX; do run_mix $m; done
fi
log "all mixes finished"

# ---------------------------------------------------------------- manifest, final evaluation, figures
AGGS=""; NAMES=""; ENSS=""; R0S=""
for m in $MIX; do
  AGGS="$AGGS runs/aggregation/m13_${m}_s$SEED"; ENSS="$ENSS runs/surrogate/plant_v2_r0s240_$m"; R0S="$R0S m13_r0_$m"
done
python scripts/build_m13_manifest.py --demo-arms runs/study/demo/arms.json --agg $AGGS \
    --ensembles $ENSS --r0-studies $R0S --out $OUT/arms.json
python scripts/run_final_evaluation.py --arms $OUT/arms.json --workers 8 --study m13_final --sets test ood > runs/logs/m13_final_eval.log 2>&1
ROUNDS_FILES=$(for m in $MIX; do echo runs/aggregation/m13_${m}_s$SEED/rounds.json; done)
python scripts/plot_sample_efficiency.py --arms $OUT/arms.json --rounds $ROUNDS_FILES --e1 "" --out _progress/figures/m13
log "M13 complete: $OUT"
