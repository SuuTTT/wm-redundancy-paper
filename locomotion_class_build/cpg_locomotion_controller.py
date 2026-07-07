#!/usr/bin/env python3
"""Parameterized Central-Pattern-Generator (CPG) locomotion controller for the
dm_control_suite LOCOMOTION class (mujoco_playground): CheetahRun, WalkerRun,
WalkerWalk, HopperHop, HopperStand.

This is the locomotion analogue of the swing-up energy-shaping controller
(pendulum_controller.py): ONE control paradigm -- a small set of coupled phase
oscillators producing rhythmic joint-target trajectories with a posture/balance
feedback term -- *instantiated per morphology* from its actuator layout.  The
controller itself is identical across tasks; only a handful of scalar/vector
parameters (freq, amplitude, phase offsets, joint centers, PD gains, posture
gains) are set per task.  That is the "uniform-within-class" claim: locomotion =
one CPG, parameterized per body, exactly as swing-up = one energy-shaper,
parameterized per pole.

DESIGN
------
* Oscillator: a single global phase  phi(t) = 2*pi*freq*t  drives all actuators;
  each actuator a_i has a phase OFFSET psi_i (legs in anti-phase, joints within a
  leg phase-locked), an AMPLITUDE amp_i, and a CENTER c_i (the nominal posture
  joint angle).  Target angle:  q*_i = c_i + amp_i * sin(phi + psi_i).
  This is the standard "coupled oscillator -> joint trajectory" CPG with the
  couplings already solved to a fixed phase lattice (constant offsets), which is
  the steady-state of a ring of identical phase-coupled oscillators.

* Joint-space PD -> normalized torque (actuators are gear-scaled motors with
  ctrlrange [-1,1]):   u_i = clip( kp_i*(q*_i - q_i) - kd_i*qd_i , -1, 1 ).

* Posture / balance feedback: a trunk-pitch and (optionally) height term that
  biases the hip/leg targets to keep the torso upright and at hopping/running
  height.  pitch error -> uniform additive lean on the "hip-like" actuators.

* PHASE exposure for the Markov residual: we expose a 2-vector (sin phi, cos phi)
  -- the only hidden CPG state is the oscillator phase, so a residual policy that
  sees (s, sin phi, cos phi) is Markov in the augmented state.

All math is pure jnp -> jit/vmap-able inside the brax PPO env.

MORPHOLOGY LAYOUTS (verified from the mujoco models on b3060)
  cheetah: nq=9 nu=6  qpos=[rootx,rootz,rooty, bthigh,bshin,bfoot, fthigh,fshin,ffoot]
           actuated qpos/qvel idx = 3..9 ; height=qpos[1] pitch=qpos[2]
  walker : nq=9 nu=6  qpos=[rootz,rootx,rooty, r_hip,r_knee,r_ankle, l_hip,l_knee,l_ankle]
           actuated idx = 3..9 ; height=qpos[0] pitch=qpos[2]
  hopper : nq=7 nu=4  qpos=[rootx,rootz,rooty, waist,hip,knee,ankle]
           actuated idx = 3..7 ; height=qpos[1] pitch=qpos[2]
"""
import jax.numpy as jp
import numpy as np

# ----------------------------------------------------------------------------
# Per-task CPG parameterization.  These are the ONLY things that change across
# the class; the controller() function below is identical for every task.
# Each entry is a plain dict of numpy arrays / floats (host-side constants).
#
# Index conventions (actuator order == env actuator order):
#   cheetah actuators: [bthigh,bshin,bfoot, fthigh,fshin,ffoot]
#   walker  actuators: [r_hip,r_knee,r_ankle, l_hip,l_knee,l_ankle]
#   hopper  actuators: [waist,hip,knee,ankle]
#
# Phase offsets put the two "legs" of cheetah/walker in anti-phase (pi apart) and
# phase-lock the joints within a leg; hopper is a single leg so it just pumps.
# ----------------------------------------------------------------------------

_PI = np.pi


def _layout(task):
  """Return (n_act, act_qpos_idx, act_qvel_idx, height_idx, pitch_idx, morph)."""
  if task in ("CheetahRun",):
    return dict(n=6, qidx=np.arange(3, 9), vidx=np.arange(3, 9),
                hidx=1, pidx=2, morph="cheetah")
  if task in ("WalkerRun", "WalkerWalk", "WalkerStand"):
    return dict(n=6, qidx=np.arange(3, 9), vidx=np.arange(3, 9),
                hidx=0, pidx=2, morph="walker")
  if task in ("HopperHop", "HopperStand"):
    return dict(n=4, qidx=np.arange(3, 7), vidx=np.arange(3, 7),
                hidx=1, pidx=2, morph="hopper")
  raise ValueError(task)


# Default per-task parameter dicts.  Tunable knobs: freq, amp (per-act), psi
# (per-act phase offset), center (per-act nominal angle), kp/kd (PD), and posture
# gains k_pitch / k_height with a target height/pitch.  Defaults below are a
# reasonable hand-init; validate_controller.py grid/CMA-tunes freq/amp/kp per task.
DEFAULTS = {
    # CHEETAH: gallop -- back legs anti-phase to front legs; thigh leads, shin/foot
    # follow.  Centers ~ mid-range crouch.  No height feedback (cheetah runs low).
    "CheetahRun": dict(
        freq=4.0,
        amp=np.array([0.6, 0.5, 0.4, 0.6, 0.5, 0.4]),
        psi=np.array([0.0, 0.5, 1.0, _PI, _PI + 0.5, _PI + 1.0]),
        center=np.array([0.0, 0.0, -0.4, -0.3, -0.2, 0.0]),
        kp=np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        kd=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        k_pitch=0.0, k_height=0.0, target_pitch=0.0, target_height=0.0,
        pitch_acts=np.array([0, 3]),  # the two thigh/hip actuators
    ),
    # WALKER: alternating gait -- right leg anti-phase to left.  Hip leads,
    # knee/ankle follow.  Strong upright posture feedback (must stay tall).
    "WalkerRun": dict(
        freq=2.2,
        amp=np.array([0.7, 0.6, 0.3, 0.7, 0.6, 0.3]),
        psi=np.array([0.0, 0.4, 0.8, _PI, _PI + 0.4, _PI + 0.8]),
        center=np.array([0.0, -0.5, 0.0, 0.0, -0.5, 0.0]),
        kp=np.array([0.9, 0.9, 0.7, 0.9, 0.9, 0.7]),
        kd=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        k_pitch=1.5, k_height=0.0, target_pitch=0.0, target_height=1.2,
        pitch_acts=np.array([0, 3]),
    ),
    "WalkerWalk": dict(
        freq=1.3,
        amp=np.array([0.55, 0.5, 0.25, 0.55, 0.5, 0.25]),
        psi=np.array([0.0, 0.4, 0.8, _PI, _PI + 0.4, _PI + 0.8]),
        center=np.array([0.0, -0.4, 0.0, 0.0, -0.4, 0.0]),
        kp=np.array([0.9, 0.9, 0.7, 0.9, 0.9, 0.7]),
        kd=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        k_pitch=1.8, k_height=0.0, target_pitch=0.0, target_height=1.2,
        pitch_acts=np.array([0, 3]),
    ),
    "WalkerStand": dict(  # degenerate CPG: amp~0, pure posture hold
        freq=0.0,
        amp=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        center=np.array([0.0, -0.3, 0.0, 0.0, -0.3, 0.0]),
        psi=np.zeros(6),
        kp=np.array([1.0, 1.0, 0.8, 1.0, 1.0, 0.8]),
        kd=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        k_pitch=2.5, k_height=0.0, target_pitch=0.0, target_height=1.2,
        pitch_acts=np.array([0, 3]),
    ),
    # HOPPER: single leg -- pump knee+ankle in phase (extend together) to bounce,
    # hip stabilizes, waist holds.  Hopping needs strong height/posture term.
    "HopperHop": dict(
        freq=3.0,
        amp=np.array([0.0, 0.3, 0.8, 0.5]),   # waist,hip,knee,ankle
        psi=np.array([0.0, 0.0, 0.0, 0.3]),
        center=np.array([0.0, -0.3, 1.2, 0.0]),
        kp=np.array([0.8, 0.8, 1.0, 0.8]),
        kd=np.array([0.05, 0.05, 0.05, 0.05]),
        k_pitch=1.5, k_height=0.0, target_pitch=0.0, target_height=0.9,
        pitch_acts=np.array([1]),  # hip
    ),
    "HopperStand": dict(  # hold a tall posture; tiny/zero oscillation
        freq=0.0,
        amp=np.array([0.0, 0.0, 0.0, 0.0]),
        psi=np.zeros(4),
        center=np.array([0.0, -0.2, 0.9, 0.0]),
        kp=np.array([0.8, 1.0, 1.0, 0.8]),
        kd=np.array([0.1, 0.1, 0.1, 0.1]),
        k_pitch=2.0, k_height=0.0, target_pitch=0.0, target_height=0.9,
        pitch_acts=np.array([1]),
    ),
}


def make_controller(task, params=None):
  """Build a jittable CPG controller closure for `task`.

  Returns ctrl_fn(qpos, qvel, t) -> (u in [-1,1]^n_act, phase2=(sin phi, cos phi)).
  `params` overrides any DEFAULTS[task] keys (used by the per-task tuner).
  All per-task constants are baked into the closure as jnp arrays.
  """
  lay = _layout(task)
  p = dict(DEFAULTS[task])
  if params:
    p.update(params)

  qidx = jp.array(lay["qidx"])
  vidx = jp.array(lay["vidx"])
  hidx = lay["hidx"]
  pidx = lay["pidx"]
  n = lay["n"]

  freq = jp.asarray(float(p["freq"]))
  amp = jp.asarray(np.asarray(p["amp"], dtype=np.float32))
  psi = jp.asarray(np.asarray(p["psi"], dtype=np.float32))
  center = jp.asarray(np.asarray(p["center"], dtype=np.float32))
  kp = jp.asarray(np.asarray(p["kp"], dtype=np.float32))
  kd = jp.asarray(np.asarray(p["kd"], dtype=np.float32))
  k_pitch = jp.asarray(float(p["k_pitch"]))
  k_height = jp.asarray(float(p["k_height"]))
  target_pitch = jp.asarray(float(p["target_pitch"]))
  target_height = jp.asarray(float(p["target_height"]))
  pitch_mask = np.zeros(n, dtype=np.float32)
  pitch_mask[np.asarray(p["pitch_acts"])] = 1.0
  pitch_mask = jp.asarray(pitch_mask)

  TWO_PI = 2.0 * np.pi

  def ctrl_fn(qpos, qvel, t):
    # current actuated joint angles / velocities
    q = qpos[..., qidx]
    qd = qvel[..., vidx]

    # t may be a scalar () or per-env (batch,).  phi0 is the scalar phase per
    # env; phi adds a trailing actuator axis so (phi0 + psi) broadcasts to
    # (..., n_act).
    phi0 = TWO_PI * freq * jp.asarray(t)          # shape (...,)
    phi = phi0[..., None]                          # shape (..., 1)
    sinp = jp.sin(phi + psi)                        # (..., n_act)
    q_des = center + amp * sinp                    # CPG target trajectory

    # posture feedback: lean the hip-like actuators to correct trunk pitch + add
    # an extension bias if torso is below target height (gated by k_height).
    pitch = qpos[..., pidx]
    height = qpos[..., hidx]
    lean = k_pitch * (pitch - target_pitch)       # >0 => pitched forward
    hbias = k_height * (target_height - height)    # >0 => too low, extend
    # lean/hbias are scalar-per-env (shape (...,)); add a trailing actuator axis
    # so they broadcast against pitch_mask (shape (n_act,)).
    posture = (lean + hbias)[..., None] * pitch_mask
    q_des = q_des + posture

    u = kp * (q_des - q) - kd * qd
    u = jp.clip(u, -1.0, 1.0)

    # phase2 uses the scalar per-env phase phi0 -> shape (..., 2).
    phase2 = jp.stack([jp.sin(phi0), jp.cos(phi0)], axis=-1)
    return u, phase2

  ctrl_fn.n_act = n
  ctrl_fn.task = task
  return ctrl_fn


# Quick self-test of shapes when run directly.
if __name__ == "__main__":
  for tk in ["CheetahRun", "WalkerRun", "WalkerWalk", "HopperHop", "HopperStand"]:
    lay = _layout(tk)
    nq = 9 if lay["morph"] != "hopper" else 7
    f = make_controller(tk)
    u, ph = f(jp.zeros(nq), jp.zeros(nq), jp.asarray(0.123))
    print(tk, "n_act", f.n_act, "u.shape", u.shape, "phase", ph.shape,
          "u", np.asarray(u).round(3))
