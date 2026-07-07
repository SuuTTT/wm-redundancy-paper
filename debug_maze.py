import sys
sys.path.insert(0, "/root/helios-rl/hjepa_navctrl")
import jax, jax.numpy as jnp, numpy as np
from pointmaze_jax import PointMaze, WALLS, _seg_blocks, _reset_one, _step_one

env = PointMaze()
# single env trace, no jit
st = jax.vmap(_reset_one)(jax.random.split(jax.random.PRNGKey(1), 1))
print("start pos", np.asarray(st.pos)[0], "goal", np.asarray(st.goal)[0])
print("WALLS\n", np.asarray(WALLS))

# manually drive the intended route with small steps and report collisions
pos = np.asarray(st.pos)[0].copy()
goal = np.asarray(st.goal)[0].copy()
waypoints = [np.array([0.80,0.30]), np.array([0.80,0.90]), goal]
import jax.numpy as jnp
seg_block = jax.jit(_seg_blocks)
cur = pos.copy()
blocked_count = 0
total = 0
for wp in waypoints:
    for _ in range(2000):
        d = wp - cur
        if np.linalg.norm(d) < 0.02:
            break
        step = np.clip(d*50, -1,1)*0.03
        nxt = np.clip(cur+step, 0,1)
        b = bool(seg_block(jnp.array(cur), jnp.array(nxt), WALLS))
        total += 1
        if b:
            blocked_count += 1
            # try to slide: stay
            cur = cur
            break
        cur = nxt
    print(f"reached toward {wp}: cur={cur} blocked_so_far={blocked_count}")
print("final", cur, "dist to goal", np.linalg.norm(cur-goal))
