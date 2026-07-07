#!/usr/bin/env python3
"""Eval every brax-PPO checkpoint of a Residual<Task> locomotion run under
Protocol A (n parallel envs, 1000-step episode, mean true task return) -> learning
curve JSON.  Same protocol as the baseline CSVs.

Usage: eval_residual_curve.py --task CheetahRun --ckpt_root <logdir>/<exp>/checkpoints --out x.json
"""
import os, sys, json, argparse, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/helios-rl/exp/tdmpc_glass/baselines_ppo_sac")
import jax_compat  # noqa
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground._src import dm_control_suite
from mujoco_playground import wrapper
import residual_locomotion as RL


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--ckpt_root", required=True)
  ap.add_argument("--n", type=int, default=128)
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=1000)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  envname = RL.register_one(args.task)

  from brax.training.agents.ppo import checkpoint as C
  from brax.training.agents.ppo import networks as N
  load_policy = lambda p: C.load_policy(p, network_factory=N.make_ppo_networks,
                                        deterministic=True)

  env = dm_control_suite.load(envname, config_overrides={"impl": "jax"})
  wenv = wrapper.wrap_for_brax_training(env, episode_length=args.steps,
                                        action_repeat=1)
  reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)

  ckpts = [os.path.abspath(d) for d in glob.glob(os.path.join(args.ckpt_root, "*"))
           if os.path.isdir(d) and os.path.basename(d).isdigit()]
  ckpts.sort(key=lambda d: int(os.path.basename(d)))
  print(f"Found {len(ckpts)} checkpoints under {args.ckpt_root}", flush=True)

  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)
  curve = []
  for cp in ckpts:
    st_steps = int(os.path.basename(cp))
    t0 = time.time()
    infer = load_policy(cp)
    jinfer = jax.jit(infer)

    def run(keys):
      state = reset(keys)
      def body(carry, _):
        state, ret = carry
        act, _ = jinfer(state.obs, jax.random.PRNGKey(0))
        ns = step(state, act)
        return (ns, ret + ns.reward), None
      (_, ret), _ = jax.lax.scan(body, (state, jp.zeros(args.n)), None,
                                 length=args.steps)
      return ret
    ret = np.array(jax.jit(run)(keys))
    curve.append({"step": st_steps, "return": float(ret.mean()),
                  "std": float(ret.std()), "min": float(ret.min())})
    print(f"  step={st_steps:>10} return={ret.mean():8.1f} "
          f"std={ret.std():7.1f} ({time.time()-t0:.1f}s)", flush=True)
    json.dump({"task": args.task, "ckpt_root": args.ckpt_root,
               "alpha": RL.RES_ALPHA, "n": args.n, "steps": args.steps,
               "curve": curve}, open(args.out, "w"), indent=2)

  if curve:
    peak = max(c["return"] for c in curve)
    fin = curve[-1]["return"]
    print(f"\n{args.task} PEAK={peak:.1f} FINAL={fin:.1f} -> {args.out}")


if __name__ == "__main__":
  main()
