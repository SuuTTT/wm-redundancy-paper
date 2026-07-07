#!/usr/bin/env bash
# A2 novelty-MPPI sweep driver. Runs tasks x arms x seeds on GPU 0,1 (2 jobs/GPU).
# Writes per-run JSONL (incremental) + per-run summary JSON (on finish) + DONE marker.
set -u
REPO=/root/helios-rl
ROOT=$REPO/exp/A2_novelty_mppi
mkdir -p "$ROOT/logs" "$ROOT/jsonl" "$ROOT/runs"
cd "$REPO" || exit 1
source .venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo
export MUJOCO_GL=egl MJPG_IMPL=jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export TDMPC_EVAL_INTERVAL=50000

STEPS=${STEPS:-300000}
SEEDS=${SEEDS:-"1 2"}
# ordered by likelihood of clean separation / feasibility of discovery
TASKS=${TASKS:-"CartpoleSwingupSparse PendulumSwingup BallInCup AcrobotSwingupSparse FingerTurnHard"}
GPUS=(0 1)
NPAR=${NPAR:-4}

declare -A ARMS
ARMS[vanilla]="NOVELTY_BETA=0"
ARMS[rnd_b0.5]="NOVELTY_TYPE=rnd NOVELTY_BETA=0.5 RND_COEF=1.0"
ARMS[rnd_b1.0]="NOVELTY_TYPE=rnd NOVELTY_BETA=1.0 RND_COEF=1.0"
ARMS[dis_b0.5]="NOVELTY_TYPE=disagreement NOVELTY_BETA=0.5"
ARM_ORDER="vanilla rnd_b0.5 rnd_b1.0 dis_b0.5"

echo "[A2] start $(date -u +%FT%TZ) steps=$STEPS seeds='$SEEDS' tasks='$TASKS' npar=$NPAR" >> "$ROOT/driver.log"

run_one(){
  local task=$1 arm=$2 seed=$3 gpu=$4
  local tag=${task}_${arm}_s${seed}
  local log=$ROOT/logs/${tag}.log
  local jsonl=$ROOT/jsonl/${tag}.jsonl
  rm -f "$jsonl"
  (
    export ${ARMS[$arm]}
    export A2_JSONL=$jsonl
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "[A2] launch $tag gpu=$gpu $(date -u +%FT%TZ)" >> "$ROOT/driver.log"
    python3 -u scripts/run_benchmark.py --algos tdmpc2 --tasks "$task" \
      --total_steps "$STEPS" --seed "$seed" --no_plot > "$log" 2>&1
    st=$?
    python3 "$ROOT/a2_summarize.py" "$jsonl" "$task" "$arm" "$seed" "$ROOT/runs/${tag}.json" >> "$ROOT/driver.log" 2>&1
    echo "[A2] done   $tag gpu=$gpu status=$st $(date -u +%FT%TZ)" >> "$ROOT/driver.log"
    # incremental verdict refresh so partial harvest is always current
    python3 "$ROOT/a2_verdict.py" "$ROOT" >> "$ROOT/driver.log" 2>&1
  )
}

i=0
for task in $TASKS; do
  for seed in $SEEDS; do
    for arm in $ARM_ORDER; do
      gpu=${GPUS[$(( i % 2 ))]}
      run_one "$task" "$arm" "$seed" "$gpu" &
      i=$(( i + 1 ))
      while (( $(jobs -r | wc -l) >= NPAR )); do wait -n; done
    done
  done
done
wait

python3 "$ROOT/a2_verdict.py" "$ROOT" >> "$ROOT/driver.log" 2>&1
echo "A2_SWEEP_DONE $(date -u +%FT%TZ) total_runs=$i" >> "$ROOT/driver.log"
