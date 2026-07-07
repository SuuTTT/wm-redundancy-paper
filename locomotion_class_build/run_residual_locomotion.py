#!/usr/bin/env python3
"""Register a Residual<Task> locomotion env and run the official brax
train_jax_ppo main on it.  Requires --env_name <Task>Residual.  The task is
parsed from env_name to register only that env (saves compile)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/helios-rl/exp/tdmpc_glass/baselines_ppo_sac")
import jax_compat  # noqa: F401

import residual_locomotion as RL

# find --env_name value
env_name = None
for i, a in enumerate(sys.argv):
  if a == "--env_name" and i + 1 < len(sys.argv):
    env_name = sys.argv[i + 1]
  elif a.startswith("--env_name="):
    env_name = a.split("=", 1)[1]
assert env_name and env_name.endswith("Residual"), f"bad env_name {env_name}"
task = env_name[:-len("Residual")]
RL.register_one(task)

sys.argv[0] = "/root/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
