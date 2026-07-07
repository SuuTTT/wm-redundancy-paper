#!/usr/bin/env python3
"""Validate / tune the CPG locomotion controller ALONE on the real
mujoco_playground locomotion envs, Protocol A (n parallel envs, 1000-step episode,
mean return), the same protocol used for the TD-MPC2 / PPO baselines.

Modes:
  single  TASK          -> controller-alone return (multi-seed) with DEFAULTS
  grid    TASK          -> coarse grid over (freq, amp_scale, kp) -> best params
                           written to <TASK>_cpg_best.json (the per-task
                           *parameterization* of the shared controller)

Run with VAL_N envs (default 64; >=64 for the pre-train validation gate).
"""
import os, sys, json, itertools, statistics
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.18")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import registry, wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_locomotion_controller as CPG

EPLEN = 1000
N = int(os.environ.get("VAL_N", "64"))


def build_env(task):
  env = registry.load(task, config_overrides={"impl": "jax"})
  wenv = wrapper.wrap_for_brax_training(env, episode_length=EPLEN, action_repeat=1)
  return wenv


def eval_params(wenv, task, params, seed=0):
  ctrl = CPG.make_controller(task, params)
  _step = jax.jit(wenv.step)
  _reset = jax.jit(wenv.reset)

  @jax.jit
  def rollout(key):
    st = _reset(jax.random.split(key, N))

    def body(carry, _):
      s, ep = carry
      u, _ph = ctrl(s.data.qpos, s.data.qvel, s.data.time)
      u = u.reshape(N, ctrl.n_act)
      ns = _step(s, u)
      return (ns, ep + ns.reward), None

    (_, ep), _ = jax.lax.scan(body, (st, jp.zeros(N)), None, length=EPLEN)
    return ep

  ep = np.asarray(rollout(jax.random.PRNGKey(seed)))
  return float(ep.mean()), float(ep.std()), float(ep.min())


def main():
  mode = sys.argv[1] if len(sys.argv) > 1 else "single"
  task = sys.argv[2] if len(sys.argv) > 2 else "CheetahRun"
  wenv = build_env(task)

  if mode == "grid":
    base = dict(CPG.DEFAULTS[task])
    amp0 = np.asarray(base["amp"], dtype=np.float32)
    kp0 = np.asarray(base["kp"], dtype=np.float32)
    f0 = float(base["freq"])
    # freq centred on default; amp & kp scaled multiplicatively.
    freqs = sorted(set([max(0.0, f0 * m) for m in (0.7, 1.0, 1.4)]))
    amp_sc = [0.7, 1.0, 1.4]
    kp_sc = [0.8, 1.3]
    results = []
    for f, asc, ksc in itertools.product(freqs, amp_sc, kp_sc):
      p = dict(base)
      p["freq"] = f
      p["amp"] = amp0 * asc
      p["kp"] = kp0 * ksc
      m, sd, mn = eval_params(wenv, task, p, seed=0)
      results.append((m, sd, mn, f, asc, ksc))
      print(f"freq={f:5.2f} amp*{asc:4.2f} kp*{ksc:4.2f}  ret={m:8.2f} "
            f"std={sd:7.2f} min={mn:8.2f}", flush=True)
    results.sort(reverse=True)
    print("\nTOP 5:")
    for r in results[:5]:
      print(f"  ret={r[0]:8.2f} std={r[1]:7.2f}  freq={r[3]:.2f} amp*{r[4]} kp*{r[5]}")
    best = results[0]
    out = {
        "task": task, "best_ret_n%d" % N: best[0], "best_std": best[1],
        "best_min": best[2], "freq": best[3], "amp_scale": best[4],
        "kp_scale": best[5],
        # materialized absolute params for the residual env:
        "amp": (amp0 * best[4]).tolist(), "kp": (kp0 * best[5]).tolist(),
    }
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"{task}_cpg_best.json")
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")
  else:
    rets = []
    # if a tuned best exists, use it; else DEFAULTS
    bestfn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"{task}_cpg_best.json")
    params = None
    if os.path.exists(bestfn):
      b = json.load(open(bestfn))
      params = {"freq": b["freq"], "amp": np.asarray(b["amp"]),
                "kp": np.asarray(b["kp"])}
      print(f"using tuned params from {bestfn}: freq={b['freq']} "
            f"amp_scale={b['amp_scale']} kp_scale={b['kp_scale']}")
    for sd in range(3):
      m, s, mn = eval_params(wenv, task, params, seed=sd)
      rets.append(m)
      print(f"seed={sd}  ret={m:8.2f} std={s:7.2f} min={mn:8.2f}", flush=True)
    out = {"task": task, "n": N, "rets": rets,
           "mean": statistics.mean(rets), "std": statistics.pstdev(rets),
           "tuned": params is not None}
    print(f"\nCPG-alone {task}: mean={out['mean']:.2f} std={out['std']:.2f} "
          f"(n={N}, {len(rets)} seeds)")
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"{task}_cpg_alone.json")
    json.dump(out, open(fn, "w"), indent=2)
    print(f"wrote {fn}")


if __name__ == "__main__":
  main()
