"""Minimal JAX 2D point-maze with a brax-like state interface.

Purpose: an EASY-low-level, LONG-HORIZON goal-reaching task to test whether
H-JEPA's hierarchical latent planning crosses competence where the low-level
primitive is TRIVIAL. The LL primitive is "move toward a (sub)goal": the action
IS a clipped velocity, so reaching a nearby goal is one-step easy. The challenge
is distance / temporal extent, not the contact primitive.

Interface mirrors brax/mujoco_playground enough for run_hjepa to reuse its
jitted lax.scan rollout:
  - reset(keys) -> State            (keys: (N,2) PRNGKey array, vmapped)
  - step(state, action) -> State
  - State has .obs (N,4), .reward (N,), .done (N,), .metrics["at_goal"], .info["dist"]
obs = [pos_x, pos_y, goal_x, goal_y]  -> a flat goal-conditioned policy is feasible.

Configurable (constructor args):
  - walls: "none" (open arena) or "c" (C-shaped detour: central barrier w/ a gap)
  - goal_radius: success tolerance
  - reward: "dense" (-dist + bonus) or "sparse" (bonus only)
  - max_speed, episode_len

The OPEN-ARENA variant is the clean positive control: long-horizon (goal far,
500 steps) but with a non-deceptive monotone-approachable reward, so a flat
goal-conditioned RL agent provably can solve it -> failure of H-JEPA there would
isolate the hierarchy, not the env.
"""
from __future__ import annotations
import flax.struct as struct
import jax
import jax.numpy as jnp


@struct.dataclass
class State:
    pos: jax.Array
    goal: jax.Array
    t: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict
    info: dict


# C-shaped detour: central horizontal barrier y=0.5, x in [0, 0.75], gap on right.
WALLS_C = jnp.array([[0.00, 0.50, 0.75, 0.50]], dtype=jnp.float32)
WALLS_NONE = jnp.zeros((1, 4), dtype=jnp.float32)   # zero-length seg -> never blocks

# Back-compat module constants (used by sanity_maze.py / run_hjepa_nav.py imports).
MAX_SPEED = 0.03
GOAL_RADIUS = 0.06
EPISODE_LEN = 500
WALLS = WALLS_C


def _seg_blocks(p0, p1, walls):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def one_wall(w):
        q0 = w[:2]; q1 = w[2:]
        d1 = cross(p0, p1, q0); d2 = cross(p0, p1, q1)
        d3 = cross(q0, q1, p0); d4 = cross(q0, q1, p1)
        return ((d1 * d2) < 0) & ((d3 * d4) < 0)

    return jnp.any(jax.vmap(one_wall)(walls))


def _sample_pos(key, lo, hi):
    return jax.random.uniform(key, (2,), minval=lo, maxval=hi)


def _make_reset(walls_is_c):
    def _reset_one(key):
        k1, k2 = jax.random.split(key)
        if walls_is_c:
            # start bottom-left (below barrier), goal top-left (above barrier).
            pos = _sample_pos(k1, jnp.array([0.05, 0.05]), jnp.array([0.45, 0.40]))
            goal = _sample_pos(k2, jnp.array([0.05, 0.65]), jnp.array([0.45, 0.95]))
        else:
            # open arena, LONG-HORIZON: start bottom-left corner, goal top-right
            # corner -> distance ~0.8-1.3 (>=~30 steps at the speed cap, well within
            # the 500-step horizon). Non-deceptive (no walls) so flat RL can solve it.
            pos = _sample_pos(k1, jnp.array([0.05, 0.05]), jnp.array([0.30, 0.30]))
            goal = _sample_pos(k2, jnp.array([0.70, 0.70]), jnp.array([0.95, 0.95]))
        obs = jnp.concatenate([pos, goal])
        dist = jnp.linalg.norm(pos - goal)
        return State(pos=pos, goal=goal, t=jnp.float32(0), obs=obs,
                     reward=jnp.float32(0.0), done=jnp.float32(0.0),
                     metrics={"at_goal": jnp.float32(0.0)}, info={"dist": dist})
    return _reset_one


def _make_step(walls, max_speed, goal_radius, episode_len, reward_mode):
    def _step_one(state, action):
        a = jnp.clip(action, -1.0, 1.0) * max_speed       # action -> velocity (trivial LL)
        new_pos = jnp.clip(state.pos + a, 0.0, 1.0)
        blocked = _seg_blocks(state.pos, new_pos, walls)  # cancel move if it crosses a wall
        new_pos = jnp.where(blocked, state.pos, new_pos)
        dist = jnp.linalg.norm(new_pos - state.goal)
        at_goal = (dist < goal_radius).astype(jnp.float32)
        if reward_mode == "sparse":
            reward = at_goal                              # +1 only at goal
        else:
            reward = -dist + 10.0 * at_goal               # dense shaping + bonus
        t = state.t + 1.0
        done = jnp.maximum(at_goal, (t >= episode_len).astype(jnp.float32))
        obs = jnp.concatenate([new_pos, state.goal])
        return state.replace(pos=new_pos, t=t, obs=obs, reward=reward, done=done,
                             metrics={"at_goal": at_goal}, info={"dist": dist})
    return _step_one


class PointMaze:
    """Vectorised (vmapped) point-maze with brax-like reset/step."""
    def __init__(self, walls="c", goal_radius=GOAL_RADIUS, reward="dense",
                 max_speed=MAX_SPEED, episode_len=EPISODE_LEN):
        self.observation_size = 4
        self.action_size = 2
        self.episode_len = episode_len
        wmap = {"c": WALLS_C, "none": WALLS_NONE}
        W = wmap[walls]
        self._reset_one = _make_reset(walls == "c")
        self._step_one = _make_step(W, max_speed, goal_radius, episode_len, reward)

    def reset(self, keys):
        return jax.vmap(self._reset_one)(keys)

    def step(self, state, action):
        nxt = jax.vmap(self._step_one)(state, action)

        def maybe_reset(s_done, s_next, key):
            fresh = self._reset_one(key)
            return jax.tree.map(lambda a, b: jnp.where(s_done > 0.5, a, b), fresh, s_next)

        N = action.shape[0]
        keys = jax.vmap(lambda i: jax.random.fold_in(
            jax.random.PRNGKey(0), (i * 99991 + (nxt.t.sum() * 7).astype(jnp.int32))))(jnp.arange(N))
        out = jax.vmap(maybe_reset)(nxt.done, nxt, keys)
        out = out.replace(reward=nxt.reward, done=nxt.done,
                          metrics=nxt.metrics, info=nxt.info)
        return out
