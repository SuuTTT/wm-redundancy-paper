#!/usr/bin/env bash
# C_hier_new driver: {flat, feudal} x {corridor, fourroom} x seeds{0,1,2}, matched budget.
# 2 GPU slots (0,1) ONLY. Per-run JSON written by feudal_maze.py. DONE marker at end.
# NO self-watcher. Parent harvests.
set -u
EXP=/root/tdmpc_glass/exp
OUT=$EXP/C_hier_new
LOGS=$OUT/logs
mkdir -p "$OUT/runs" "$LOGS"
VENV=/root/tdmpc_glass/venv
COMMON="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25"
GPUS=(0 1)
STEPS=${STEPS:-400000}

# Build queue (interleave arms/mazes so partial harvest is balanced)
Q=()
# Difficulty gradient of open rooms (isolates long-horizon sparse credit from wall-nav):
#   room 7.07u | midroom 9.9u | bigroom 12.73u. seed-outer => n=1 everywhere first.
for s in 0 1 2; do
  for maze in room midroom bigroom; do
    for arm in flat feudal; do
      Q+=("--arm $arm --maze $maze --seed $s --steps $STEPS --out_dir $OUT/runs")
    done
  done
done
echo "[driver] queue ${#Q[@]} runs, STEPS=$STEPS, start $(date)" | tee -a "$LOGS/driver.log"

gpu_busy() {  # 0 => a compute proc of ours is running on gpu $1
  local g=$1
  local apps; apps=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' \n')
  [ -n "$apps" ]
}
launch() {
  local g=$1; shift
  local args="$1"
  local name; name=$(echo "$args" | sed -E 's/.*--arm ([a-z]+) --maze ([a-z]+) --seed ([0-9]+).*/\1_\2_s\3/')
  echo "[driver $(date +%H:%M:%S)] GPU$g <- $name" | tee -a "$LOGS/driver.log"
  setsid bash -c "cd $EXP && source $VENV/bin/activate && env $COMMON CUDA_VISIBLE_DEVICES=$g python feudal_maze.py $args >> '$LOGS/$name.log' 2>&1" </dev/null &
}

qi=0
while [ "$qi" -lt "${#Q[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$qi" -ge "${#Q[@]}" ] && break
    if ! gpu_busy "$g"; then
      launch "$g" "${Q[$qi]}"
      qi=$((qi+1))
      sleep 40   # let JIT/alloc settle before checking the other GPU
    fi
  done
  sleep 30
done

# wait for both GPUs to drain (only my jobs run on 0/1)
while gpu_busy 0 || gpu_busy 1; do sleep 30; done
echo "[driver] all runs launched+drained qi=$qi $(date)" | tee -a "$LOGS/driver.log"
# aggregate -> summary.json + VERDICT.md (all numbers from runs/*.json)
source "$VENV/bin/activate" && python "$EXP/agg_C_hier.py" >> "$LOGS/driver.log" 2>&1 || echo "[driver] agg failed" | tee -a "$LOGS/driver.log"
touch "$OUT/DONE"
echo "[driver] DONE marker written $(date)" | tee -a "$LOGS/driver.log"
