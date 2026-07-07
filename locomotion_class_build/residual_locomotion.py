#!/usr/bin/env python3
"""Abstraction-in-the-loop residual envs for the LOCOMOTION class, mirroring the
pendulum_abstraction / hl_pickcube residual pattern but for the DMC locomotion
tasks driven by the shared CPG controller (cpg_locomotion_controller).

Executed control (what the base env .step receives):
    u_t = clip( u_cpg(qpos_t, qvel_t, t) + alpha * pi_res(obs_aug_t), -1, 1 )

  * u_cpg  = the parameterized CPG controller for this task (per-morphology
             instantiation of the shared paradigm), PURE state+time feedback.
  * pi_res = brax PPO policy output in [-1,1]^n_act.
  * alpha  = fixed authority (env var RES_ALPHA, default 1.0).
  * obs_aug= base task obs AUGMENTED with the CPG phase (sin phi, cos phi) so the
             residual is Markov in (s, phase) -- the CPG's only hidden state is
             its oscillator phase.

Reward is UNCHANGED (the env's true task reward), so selection/eval is on the
true task metric, never a shaped proxy.

Per task we load the tuned CPG params from <TASK>_cpg_best.json if present (the
per-task parameterization produced by validate_controller.py grid), else DEFAULTS.
"""
import os, json
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import dm_control_suite
from mujoco_playground._src.dm_control_suite import cheetah as _cheetah
from mujoco_playground._src.dm_control_suite import walker as _walker
from mujoco_playground._src.dm_control_suite import hopper as _hopper

import cpg_locomotion_controller as CPG

RES_ALPHA = float(os.environ.get("RES_ALPHA", "1.0"))
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_tuned(task):
  fn = os.path.join(_HERE, f"{task}_cpg_best.json")
  if os.path.exists(fn):
    b = json.load(open(fn))
    return {"freq": b["freq"], "amp": np.asarray(b["amp"], dtype=np.float32),
            "kp": np.asarray(b["kp"], dtype=np.float32)}
  return None


# Map task -> (base_class, ctor_kwargs, default_config)
_SPECS = {
    "CheetahRun": (_cheetah.Run, {}, _cheetah.default_config),
    "WalkerRun": (_walker.PlanarWalker, {"move_speed": _walker.RUN_SPEED},
                  _walker.default_config),
    "WalkerWalk": (_walker.PlanarWalker, {"move_speed": _walker.WALK_SPEED},
                   _walker.default_config),
    "HopperHop": (_hopper.Hopper, {"hopping": True}, _hopper.default_config),
    "HopperStand": (_hopper.Hopper, {"hopping": False}, _hopper.default_config),
}


def make_residual_class(task):
  base_cls, ctor_kwargs, _ = _SPECS[task]
  tuned = _load_tuned(task)
  ctrl = CPG.make_controller(task, tuned)
  n_act = ctrl.n_act
  alpha = RES_ALPHA

  class ResidualLoco(base_cls):
    def __init__(self, config=None, config_overrides=None):
      kw = dict(ctor_kwargs)
      if config is not None:
        kw["config"] = config
      if config_overrides is not None:
        kw["config_overrides"] = config_overrides
      super().__init__(**kw)
      self._ctrl = ctrl
      self._alpha = alpha

    @property
    def observation_size(self):
      # Our reset/step return the AUGMENTED obs (base + 2-dim phase), so just
      # report the shape of our own reset's obs. (Do NOT call super() and add 2:
      # super().observation_size would invoke self.reset -- already augmented --
      # and double-count.)
      import jax as _jax
      abstract = _jax.eval_shape(self.reset, _jax.random.PRNGKey(0))
      return abstract.obs.shape[-1]

    def _aug(self, state):
      _u, phase2 = self._ctrl(state.data.qpos, state.data.qvel, state.data.time)
      return jp.concatenate([state.obs, phase2], axis=-1)

    def reset(self, rng):
      state = super().reset(rng)
      return state.replace(obs=self._aug(state))

    def step(self, state, action):
      u_cpg, _ph = self._ctrl(state.data.qpos, state.data.qvel, state.data.time)
      u_cpg = u_cpg.reshape(action.shape)
      u_exec = jp.clip(u_cpg + self._alpha * action, -1.0, 1.0)
      ns = super().step(state, u_exec)
      return ns.replace(obs=self._aug(ns))

  ResidualLoco.__name__ = f"Residual{task}"
  return ResidualLoco


def register_all():
  for task in _SPECS:
    cls = make_residual_class(task)
    cfg = _SPECS[task][2]
    dm_control_suite.register_environment(f"{task}Residual", cls, cfg)


def register_one(task):
  cls = make_residual_class(task)
  cfg = _SPECS[task][2]
  dm_control_suite.register_environment(f"{task}Residual", cls, cfg)
  return f"{task}Residual"
