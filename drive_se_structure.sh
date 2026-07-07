#!/bin/bash
# Thread-D SE-as-STRUCTURE sweep driver.
# Arms: none (baseline) | se_contrastive | se_tree  -- FIXED lambda=0.03 (NOT grad-matched;
#   0.03 chosen on a WalkerWalk pilot purely to keep l_pred near its baseline value).
# Grid: 3 arms x 3 tasks x 2 seeds = 18 runs, 12000 steps each.
# GPUs 2,3 only (leave 0,1 for the other agent). <=2 concurrent per GPU (~2.7GB each).
# Per-run JSONs written by se_structure.py; DONE marker at the end. NO self-watcher.
set -u
cd /root/tdmpc_glass/exp/D_se_structure
PY=/root/tdmpc_glass/venv/bin/python
OUT=runs
mkdir -p $OUT logs
LAM=0.03
STEPS=12000
GPUS=(2 3)
MAXPER=2                      # max concurrent jobs per GPU
declare -A NRUN=( [2]=0 [3]=0 )

JOBS=()
for task in WalkerWalk CheetahRun ReacherEasy; do
  for seed in 0 1; do
    for arm in none se_contrastive se_tree; do
      JOBS+=("$task $arm $seed")
    done
  done
done

launch () {
  local task=$1 arm=$2 seed=$3 gpu=$4
  local tag="${task}_${arm}_seed${seed}"
  # skip if already finished
  if [ -f "$OUT/run_${task}_${arm}_seed${seed}.json" ]; then
    echo "SKIP $tag (exists)"; return
  fi
  echo "LAUNCH $tag on gpu$gpu"
  $PY se_structure.py train --task $task --cond $arm --seed $seed --gpu $gpu \
      --steps $STEPS --lam $LAM --outdir $OUT > logs/${tag}.log 2>&1
  echo "FINISH $tag exit=$?"
}

# strict per-GPU slot accounting: PID -> gpu
declare -A PIDGPU
free_gpu () {                       # echo a gpu with a free slot, else empty
  for g in "${GPUS[@]}"; do
    local c=0
    for p in "${!PIDGPU[@]}"; do
      [ "${PIDGPU[$p]}" = "$g" ] && kill -0 "$p" 2>/dev/null && c=$((c+1))
    done
    if [ $c -lt $MAXPER ]; then echo "$g"; return; fi
  done
}
reap () {                           # drop finished pids from the table
  for p in "${!PIDGPU[@]}"; do kill -0 "$p" 2>/dev/null || unset 'PIDGPU[$p]'; done
}

for job in "${JOBS[@]}"; do
  read task arm seed <<< "$job"
  gpu=$(free_gpu)
  while [ -z "$gpu" ]; do wait -n; reap; gpu=$(free_gpu); done
  launch "$task" "$arm" "$seed" "$gpu" &
  PIDGPU[$!]=$gpu
done
wait
echo "ALL_RUNS_DONE $(date -u +%FT%TZ)" > $OUT/DONE
echo "ALL_RUNS_DONE"
