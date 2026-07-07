#!/bin/bash
# Foreground variant: run ONE Residual<Task> PPO training and BLOCK until done.
# Used by orchestrate.sh's GPU pool (it handles parallelism). NEVER --save_full_state.
set -u
cd /root/helios-rl/exp/tdmpc_glass/locomotion_class
PY=/root/helios-rl/.venv/bin/python3
TASK=${TASK:?set TASK}
SEED=${SEED:-1}
GPU=${GPU:-0}
STEPS=${STEPS:-8000000}
NEVALS=${NEVALS:-16}
NENVS=${NENVS:-1024}
ALPHA=${ALPHA:-1.0}
MEMFRAC=${MEMFRAC:-0.9}

ld=logs/${TASK}_res_a${ALPHA}_s${SEED}
rm -rf "$ld"; mkdir -p "$ld"
echo "[$(date +%H:%M:%S)] FG train task=$TASK seed=$SEED gpu=$GPU steps=$STEPS -> $ld"
RES_ALPHA=$ALPHA CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC \
MUJOCO_GL=egl $PY run_residual_locomotion.py \
  --env_name ${TASK}Residual --impl jax \
  --num_timesteps $STEPS --num_evals $NEVALS --num_envs $NENVS --seed $SEED \
  --logdir "$ld" > "$ld.log" 2>&1
echo "[$(date +%H:%M:%S)] FG done task=$TASK seed=$SEED rc=$?"
