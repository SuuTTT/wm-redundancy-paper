"""Sanity check: (1) a SCRIPTED waypoint controller solves the maze (proves the
detour route exists and LL=velocity is trivially sufficient), and (2) a naive
greedy 'go straight at goal' controller FAILS (proves the task needs routing,
i.e. it's genuinely long-horizon / not solvable by a reflex). If scripted
solves and greedy fails, the env is a fair hierarchy test bed."""
import sys
sys.path.insert(0, "/root/helios-rl/hjepa_navctrl")
import jax, jax.numpy as jnp, numpy as np
from pointmaze_jax import PointMaze, EPISODE_LEN, GOAL_RADIUS

WALLS = sys.argv[1] if len(sys.argv) > 1 else "c"
env = PointMaze(walls=WALLS)
N = 64
key = jax.random.PRNGKey(0)
_reset = jax.jit(env.reset)
_step = jax.jit(env.step)


def run(policy, label, stateful=False):
    st = _reset(jax.random.split(jax.random.PRNGKey(1), N))
    alive = np.ones(N)
    succ = np.zeros(N)
    steps_to_goal = np.full(N, EPISODE_LEN, np.float32)
    wp_idx = np.zeros(N, np.int32)   # for the stateful scripted router
    for t in range(EPISODE_LEN):
        if stateful:
            a, wp_idx = policy(st, wp_idx)
        else:
            a = policy(st)
        st = _step(st, a)
        ag = np.asarray(st.metrics["at_goal"])
        newly = (ag > 0.5) & (succ < 0.5) & (alive > 0.5)
        steps_to_goal[newly] = t
        succ = np.maximum(succ, np.where(alive > 0.5, ag, 0.0))
        alive = alive * (1.0 - (np.asarray(st.done) > 0.5))
        if alive.sum() == 0:
            break
    print(f"[{label}] success={succ.mean():.3f} (n={N}) "
          f"median_steps_to_goal={np.median(steps_to_goal[succ>0.5]) if succ.any() else 'NA'}")
    return succ.mean()


def greedy(st):
    # go straight at the goal (reflex; should FAIL on the detour maze)
    d = st.goal - st.pos
    return jnp.clip(d * 50.0, -1.0, 1.0)


def scripted(st, wp_idx):
    # sequential waypoint router: per-env advance through wp1 (gap, low) -> wp2
    # (gap, high) -> goal. Advance index when within tolerance of current wp.
    # Proves a velocity-LL CAN execute the C-shaped detour.
    pos = np.asarray(st.pos)
    goal = np.asarray(st.goal)
    wp1 = np.array([0.85, 0.30])
    wp2 = np.array([0.85, 0.85])
    targets = np.stack([np.broadcast_to(wp1, goal.shape),
                        np.broadcast_to(wp2, goal.shape),
                        goal], axis=0)             # (3, N, 2)
    cur_t = targets[wp_idx, np.arange(N)]          # (N, 2)
    reached = np.linalg.norm(pos - cur_t, axis=-1) < 0.04
    wp_idx = np.minimum(wp_idx + reached.astype(np.int32), 2)
    cur_t = targets[wp_idx, np.arange(N)]
    a = np.clip((cur_t - pos) * 50.0, -1.0, 1.0)
    return jnp.asarray(a), wp_idx


print("WALLS layout: central barrier y=0.5 (gap x>0.75); start bottom-left, goal top-left")
sg = run(greedy, "greedy-reflex")
ss = run(scripted, "scripted-router", stateful=True)
print()
print(f"VERDICT: env is a fair hierarchy testbed = {ss > 0.8 and sg < 0.5}  "
      f"(scripted solves={ss:.2f}>0.8, greedy fails={sg:.2f}<0.5)")
