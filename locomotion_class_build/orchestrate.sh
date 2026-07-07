#!/bin/bash
# Autonomous CPG-locomotion-class driver for b3060 (4x RTX 3060).
# Keeps all 4 GPUs busy for many hours:
#   PHASE 1  per-task CPG grid tuning (4 tasks in parallel, 1/GPU) + the 5th after
#   PHASE 2  controller-alone validation (n=64) for all 5 tasks
#   PHASE 3  residual PPO training, 5 tasks x 2 seeds = 10 runs, 4 at a time
#   PHASE 4  eval every checkpoint of every run -> eval_curve_<seed>.json
#   PHASE 5  make_verdict.py -> VERDICT.json
# NEVER --save_full_state.  All outputs read from disk.  Logs in logs/.
set -u
cd /root/helios-rl/exp/tdmpc_glass/locomotion_class
rm -rf __pycache__
PY=/root/helios-rl/.venv/bin/python3
mkdir -p logs tune_logs
TASKS=(CheetahRun WalkerRun HopperHop WalkerWalk HopperStand)
GPUS=(0 1 2 3)
STEPS=${STEPS:-10000000}
NEVALS=${NEVALS:-20}
SEEDS=(${SEEDS:-1 2 3})

log(){ echo "[$(date +%H:%M:%S)] $*"; }

run_gpu_pool() {
  # run_gpu_pool "<cmd template with {GPU} and positional task>" task1 task2 ...
  # schedules each task onto a free GPU, 4 at a time.
  local tmpl="$1"; shift
  local items=("$@")
  local -a gpu_pid=(0 0 0 0)
  local i=0
  for it in "${items[@]}"; do
    # find a free gpu slot
    while true; do
      for g in 0 1 2 3; do
        if [ "${gpu_pid[$g]}" = "0" ] || ! kill -0 "${gpu_pid[$g]}" 2>/dev/null; then
          local cmd="${tmpl//\{GPU\}/$g}"
          cmd="${cmd//\{ITEM\}/$it}"
          log "GPU$g <- $it"
          eval "$cmd" &
          gpu_pid[$g]=$!
          i=$((i+1))
          break 2
        fi
      done
      sleep 10
    done
  done
  # wait for all
  for g in 0 1 2 3; do
    [ "${gpu_pid[$g]}" != "0" ] && wait "${gpu_pid[$g]}" 2>/dev/null
  done
}

# ---------------- PHASE 1: grid tuning ----------------
log "PHASE 1: CPG grid tuning"
run_gpu_pool 'CUDA_VISIBLE_DEVICES={GPU} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 VAL_N=64 MUJOCO_GL=egl '"$PY"' validate_controller.py grid {ITEM} > tune_logs/grid_{ITEM}.log 2>&1' \
  "${TASKS[@]}"
log "PHASE 1 done; best params:"; ls -1 *_cpg_best.json 2>/dev/null

# ---------------- PHASE 2: controller-alone validation ----------------
log "PHASE 2: controller-alone validation (n=64)"
run_gpu_pool 'CUDA_VISIBLE_DEVICES={GPU} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 VAL_N=64 MUJOCO_GL=egl '"$PY"' validate_controller.py single {ITEM} > tune_logs/alone_{ITEM}.log 2>&1' \
  "${TASKS[@]}"
log "PHASE 2 done; CPG-alone:"; for t in "${TASKS[@]}"; do [ -f ${t}_cpg_alone.json ] && echo "  $t $(grep -o '\"mean\": [0-9.]*' ${t}_cpg_alone.json | head -1)"; done

# ---------------- PHASE 3: residual training (10 runs, 4 at a time) ----------------
log "PHASE 3: residual PPO training (5 tasks x 2 seeds)"
RUNS=()
for t in "${TASKS[@]}"; do for s in "${SEEDS[@]}"; do RUNS+=("$t:$s"); done; done
run_gpu_pool 'bash -c '"'"'IFS=: read t s <<< "{ITEM}"; TASK=$t SEED=$s GPU={GPU} STEPS='"$STEPS"' NEVALS='"$NEVALS"' MEMFRAC=0.9 bash launch_one_fg.sh'"'"'' \
  "${RUNS[@]}"
log "PHASE 3 done"

# ---------------- PHASE 4: eval checkpoints ----------------
log "PHASE 4: eval residual checkpoints"
EVALS=()
for t in "${TASKS[@]}"; do for s in "${SEEDS[@]}"; do EVALS+=("$t:$s"); done; done
run_gpu_pool 'bash -c '"'"'IFS=: read t s <<< "{ITEM}"; ld=logs/${t}_res_a1.0_s${s}; ck=$(find $ld -maxdepth 3 -name checkpoints -type d | head -1); [ -z "$ck" ] && { echo "no ckpts $ld"; exit 0; }; CUDA_VISIBLE_DEVICES={GPU} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 MUJOCO_GL=egl '"$PY"' eval_residual_curve.py --task $t --ckpt_root $ck --n 128 --out $ld/eval_curve_s${s}.json > tune_logs/eval_${t}_s${s}.log 2>&1'"'"'' \
  "${EVALS[@]}"
log "PHASE 4 done"

# ---------------- PHASE 5: verdict ----------------
log "PHASE 5: verdict"
$PY make_verdict.py > tune_logs/verdict.log 2>&1
cat tune_logs/verdict.log
log "ALL DONE"
