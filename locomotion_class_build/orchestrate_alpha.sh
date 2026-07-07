#!/bin/bash
# PHASE 6 (chained after orchestrate.sh): residual-AUTHORITY sweep to answer
# "is the CPG prior an anchor or does the residual dominate?" (user question b).
# alpha in {0.5, 2.0} on the 3 primary tasks x 2 seeds, 4 at a time, then eval +
# fold into VERDICT.json (make_verdict.py reads logs/<task>_res_a<alpha>_s<seed>).
# Waits for the main orchestrator to finish first.  NEVER --save_full_state.
set -u
cd /root/helios-rl/exp/tdmpc_glass/locomotion_class
PY=/root/helios-rl/.venv/bin/python3
STEPS=${STEPS:-10000000}
NEVALS=${NEVALS:-20}
TASKS=(CheetahRun WalkerRun HopperHop)
ALPHAS=(0.5 2.0)
SEEDS=(1 2)

log(){ echo "[$(date +%H:%M:%S)] [ALPHA] $*"; }

# wait for the main training+eval orchestrator to be done (its python jobs gone)
log "waiting for main orchestrator to finish..."
while pgrep -f "run_residual_locomotion.py" >/dev/null 2>&1 || \
      pgrep -f "eval_residual_curve.py" >/dev/null 2>&1; do sleep 60; done
log "main orchestrator done; starting alpha sweep"

run_pool() {
  local kind="$1"; shift
  local items=("$@")
  local -a gpu_pid=(0 0 0 0)
  for it in "${items[@]}"; do
    while true; do
      for g in 0 1 2 3; do
        if [ "${gpu_pid[$g]}" = "0" ] || ! kill -0 "${gpu_pid[$g]}" 2>/dev/null; then
          IFS=: read t a s <<< "$it"
          if [ "$kind" = train ]; then
            log "GPU$g train $t a=$a s=$s"
            ( TASK=$t SEED=$s GPU=$g ALPHA=$a STEPS=$STEPS NEVALS=$NEVALS MEMFRAC=0.9 \
              bash launch_one_fg.sh ) &
          else
            log "GPU$g eval $t a=$a s=$s"
            ( ld=logs/${t}_res_a${a}_s${s}
              ck=$(find $ld -maxdepth 3 -name checkpoints -type d | head -1)
              [ -z "$ck" ] && { echo "no ckpts $ld"; exit 0; }
              CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 MUJOCO_GL=egl \
              $PY eval_residual_curve.py --task $t --ckpt_root $ck --n 128 \
                --out $ld/eval_curve_s${s}.json > tune_logs/eval_${t}_a${a}_s${s}.log 2>&1 ) &
          fi
          gpu_pid[$g]=$!
          break 2
        fi
      done
      sleep 10
    done
  done
  for g in 0 1 2 3; do [ "${gpu_pid[$g]}" != "0" ] && wait "${gpu_pid[$g]}" 2>/dev/null; done
}

RUNS=()
for t in "${TASKS[@]}"; do for a in "${ALPHAS[@]}"; do for s in "${SEEDS[@]}"; do
  RUNS+=("$t:$a:$s"); done; done; done

log "PHASE 6a: train ${#RUNS[@]} alpha-sweep runs"
run_pool train "${RUNS[@]}"
log "PHASE 6b: eval alpha-sweep runs"
run_pool eval "${RUNS[@]}"
log "PHASE 6c: refresh verdict (now includes alpha sweep)"
$PY make_verdict.py > tune_logs/verdict_with_alpha.log 2>&1
cat tune_logs/verdict_with_alpha.log
log "ALPHA SWEEP DONE"
