#!/usr/bin/env python
"""
C_hier_new: LEARNED 2-level hierarchy vs flat, on long-horizon sparse PointMaze.

Directly addresses the hierarchy NULL: prior campaign's low level was an ANALYTIC
phase controller. Here BOTH levels are learned from scratch (TD3):
  - flat   : single TD3 agent, env (sparse) reward only.
  - feudal : HL TD3 emits a subgoal (absolute maze coord) every K env-steps;
             LL is a goal-conditioned TD3 rewarded by dense progress-to-subgoal
             (intrinsic, self-generated -- NOT privileged task info). HL trained
             on the coarse timescale with the env's sparse goal reward.

Matched budget = matched total ENV STEPS. REAL success = fraction of deterministic
eval episodes that reach the TRUE goal. Shaped/intrinsic reward is NEVER reported
as success. All numbers written to a per-run JSON, incrementally.
"""
import os, sys, json, time, argparse
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# ----------------------------- PointMaze env -----------------------------
# Continuous 2D point mass on an occupancy grid. Velocity(force)-controlled.
# obs given to policy = [x,y,gx,gy] normalized to [-1,1]. Sparse reward: +1 on
# reaching goal (episode ends), else 0. Long horizon via max_steps.

class PointMaze:
    """Continuous point mass in a discrete-wall maze."""
    def __init__(self, name, seed=0, max_steps=None):
        specs = {
            # OPEN room, goal at opposite corner (~7-unit diagonal). No interior walls:
            # isolates LONG-HORIZON SPARSE CREDIT from wall-navigation. Goal is beyond
            # flat's per-step-noise exploration reach but within feudal's coarse HL reach.
            "room": [
                "11111111",
                "1S000001",
                "10000001",
                "10000001",
                "10000001",
                "10000001",
                "100000G1",  # goal cell (6,6); diagonal ~7.07 units from start
                "11111111",
            ],
            # MID open room (~9.9-unit diagonal): goal beyond flat's per-step-noise
            # exploration reach + long-horizon credit, within feudal's coarse HL reach.
            "midroom": [
                "1111111111",
                "1S00000001",
                "1000000001",
                "1000000001",
                "1000000001",
                "1000000001",
                "1000000001",
                "1000000001",
                "10000000G1",  # goal cell (8,8)
                "1111111111",
            ],
            # LARGER open room (~12.7-unit diagonal): harder exploration/credit.
            "bigroom": [
                "111111111111",
                "1S0000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "100000000001",
                "1000000000G1",  # goal cell (10,10)
                "111111111111",
            ],
            "corridor": [
                "1111111",
                "1S00001",
                "1111101",
                "1000001",
                "1011111",
                "1000001",  # goal placed at (5,5) programmatically
                "1111111",
            ],
            "umaze": [
                "1111111",
                "1S000001"[:7],
                "1111101",
                "1000001",
                "1011111",
                "1G00001",
                "1111111",
            ],
            "fourroom": [
                "111111111",
                "1S000001 1".replace(" 1", "1")[:9],
                "100000001",
                "111101111",
                "100000001",
                "100000001",
                "111110111",
                "100000001",
                "1000000G1",
                "111111111",
            ],
        }
        rows = specs[name]
        H = len(rows); W = len(rows[0])
        self.grid = np.zeros((H, W), dtype=np.int8)
        self.start_cell = None; self.goal_cell = None
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                if c == '1':
                    self.grid[i, j] = 1
                elif c == 'S':
                    self.start_cell = (i, j)
                elif c == 'G':
                    self.goal_cell = (i, j)
        # fallbacks for mazes where S/G omitted: pick free cells
        if self.start_cell is None:
            self.start_cell = tuple(np.argwhere(self.grid == 0)[0])
        if self.goal_cell is None:
            free = np.argwhere(self.grid == 0)
            self.goal_cell = tuple(free[-1])
        self.H, self.W = H, W
        # world coords: cell (i,j) center at (x=j+0.5, y=(H-1-i)+0.5)
        self.start = np.array([self.start_cell[1] + 0.5, (H - 1 - self.start_cell[0]) + 0.5])
        self.goal = np.array([self.goal_cell[1] + 0.5, (H - 1 - self.goal_cell[0]) + 0.5])
        self.goal_radius = 0.5
        self.sg_radius = 0.45
        self.dt = 0.3            # max move 0.3/step < 0.5 -> cannot tunnel 1-cell walls
        self.max_speed = 1.0
        self.subgoal_range = 2.5  # HL relative-subgoal reach (~reachable within K steps)
        if max_steps is None:
            # long horizon: proportional to maze size (generous but sparse-credit)
            max_steps = 10 * (H + W)
        self.max_steps = int(max_steps)
        self.rng = np.random.RandomState(seed)
        self.scale = np.array([self.W, self.H], dtype=np.float32)
        self.pos = self.start.copy()
        self.t = 0

    def _is_wall(self, x, y):
        j = int(np.floor(x)); i = int(self.H - 1 - np.floor(y))
        if i < 0 or i >= self.H or j < 0 or j >= self.W:
            return True
        return self.grid[i, j] == 1

    def reset(self):
        # small random start jitter within the start cell for robustness
        jx = self.rng.uniform(-0.2, 0.2); jy = self.rng.uniform(-0.2, 0.2)
        self.pos = self.start + np.array([jx, jy])
        self.t = 0
        return self._obs()

    def _obs(self):
        # normalized to ~[-1,1]
        p = (self.pos / self.scale) * 2 - 1
        g = (self.goal / self.scale) * 2 - 1
        return np.concatenate([p, g]).astype(np.float32)

    def obs_for_subgoal(self, subgoal_world):
        p = (self.pos / self.scale) * 2 - 1
        s = (subgoal_world / self.scale) * 2 - 1
        return np.concatenate([p, s]).astype(np.float32)

    def subgoal_from_action(self, a):
        # a in [-1,1]^2 -> RELATIVE offset from current pos (HIRO/feudal style):
        # local, reachable-within-K waypoint => LL is reliably learnable, HL steers.
        sg = self.pos + np.asarray(a, dtype=np.float32) * self.subgoal_range
        sg[0] = np.clip(sg[0], 0.05, self.W - 0.05)
        sg[1] = np.clip(sg[1], 0.05, self.H - 0.05)
        return sg

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
        v = a * self.max_speed
        newp = self.pos + v * self.dt
        # axis-separated collision handling
        cand = self.pos.copy()
        if not self._is_wall(newp[0], self.pos[1]):
            cand[0] = newp[0]
        if not self._is_wall(cand[0], newp[1]):
            cand[1] = newp[1]
        # keep in bounds
        cand[0] = np.clip(cand[0], 0.05, self.W - 0.05)
        cand[1] = np.clip(cand[1], 0.05, self.H - 0.05)
        self.pos = cand
        self.t += 1
        dist = np.linalg.norm(self.pos - self.goal)
        success = dist <= self.goal_radius
        timeout = self.t >= self.max_steps
        rew = 1.0 if success else 0.0
        done = bool(success or timeout)
        info = {"success": bool(success), "dist": float(dist), "pos": self.pos.copy()}
        return self._obs(), rew, done, info

    def dist_to(self, world_pt):
        return float(np.linalg.norm(self.pos - world_pt))

# ----------------------------- TD3 (JAX/flax) -----------------------------
class MLP(nn.Module):
    out: int
    hidden: int = 256
    final_tanh: bool = False
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.Dense(self.out)(x)
        if self.final_tanh:
            x = nn.tanh(x)
        return x

class Critic(nn.Module):
    hidden: int = 256
    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        q1 = MLP(1, self.hidden)(x)
        q2 = MLP(1, self.hidden)(x)
        return q1, q2

class TD3:
    def __init__(self, obs_dim, act_dim, seed, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, hidden=256):
        self.obs_dim = obs_dim; self.act_dim = act_dim
        self.gamma = gamma; self.tau = tau
        self.policy_noise = policy_noise; self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        key = jax.random.PRNGKey(seed)
        self.actor = MLP(act_dim, hidden, final_tanh=True)
        self.critic = Critic(hidden)
        k1, k2, key = jax.random.split(key, 3)
        dummy_o = jnp.zeros((1, obs_dim)); dummy_a = jnp.zeros((1, act_dim))
        self.ap = self.actor.init(k1, dummy_o)
        self.atp = self.ap
        self.cp = self.critic.init(k2, dummy_o, dummy_a)
        self.ctp = self.cp
        self.aopt = optax.adam(actor_lr); self.copt = optax.adam(critic_lr)
        self.aos = self.aopt.init(self.ap); self.cos = self.copt.init(self.cp)
        self.key = key
        self._it = 0
        self._build()

    def _build(self):
        actor = self.actor; critic = self.critic
        gamma = self.gamma; tau = self.tau
        pn = self.policy_noise; nc = self.noise_clip

        @jax.jit
        def act_det(ap, o):
            return actor.apply(ap, o)

        @jax.jit
        def critic_update(cp, cos, atp, ctp, ap, batch, key):
            o, a, r, o2, d = batch
            noise = jnp.clip(jax.random.normal(key, a.shape) * pn, -nc, nc)
            a2 = jnp.clip(actor.apply(atp, o2) + noise, -1, 1)
            q1t, q2t = critic.apply(ctp, o2, a2)
            qt = jnp.minimum(q1t, q2t)
            y = r + gamma * (1.0 - d) * qt.squeeze(-1)
            def closs(cp):
                q1, q2 = critic.apply(cp, o, a)
                l = ((q1.squeeze(-1) - y) ** 2 + (q2.squeeze(-1) - y) ** 2).mean()
                return l
            l, g = jax.value_and_grad(closs)(cp)
            upd, cos = self.copt.update(g, cos)
            cp = optax.apply_updates(cp, upd)
            return cp, cos, l

        @jax.jit
        def actor_update(ap, aos, cp, batch):
            o = batch[0]
            def aloss(ap):
                a = actor.apply(ap, o)
                q1, _ = critic.apply(cp, o, a)
                return -q1.mean()
            l, g = jax.value_and_grad(aloss)(ap)
            upd, aos = self.aopt.update(g, aos)
            ap = optax.apply_updates(ap, upd)
            return ap, aos, l

        @jax.jit
        def soft_update(tp, p):
            return jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, tp, p)

        self._act_det = act_det
        self._critic_update = critic_update
        self._actor_update = actor_update
        self._soft_update = soft_update

    def act(self, o, noise=0.0):
        o = jnp.asarray(o[None])
        a = np.array(self._act_det(self.ap, o))[0]
        if noise > 0:
            a = a + np.random.randn(self.act_dim) * noise
        return np.clip(a, -1, 1)

    def update(self, batch):
        self._it += 1
        self.key, k = jax.random.split(self.key)
        b = tuple(jnp.asarray(x) for x in batch)
        self.cp, self.cos, cl = self._critic_update(self.cp, self.cos, self.atp, self.ctp, self.ap, b, k)
        al = 0.0
        if self._it % self.policy_delay == 0:
            self.ap, self.aos, al = self._actor_update(self.ap, self.aos, self.cp, b)
            self.atp = self._soft_update(self.atp, self.ap)
            self.ctp = self._soft_update(self.ctp, self.cp)
        return float(cl)

class Replay:
    def __init__(self, obs_dim, act_dim, size=int(3e5)):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size = size; self.ptr = 0; self.full = False
    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i] = o; self.a[i] = a; self.r[i] = r; self.o2[i] = o2; self.d[i] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.full = True
    def __len__(self):
        return self.size if self.full else self.ptr
    def sample(self, bs):
        n = len(self)
        idx = np.random.randint(0, n, size=bs)
        return (self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx])

# ----------------------------- eval -----------------------------
def eval_flat(env, agent, episodes=50):
    succ = 0; dists = []
    for _ in range(episodes):
        o = env.reset(); done = False
        while not done:
            a = agent.act(o, noise=0.0)
            o, r, done, info = env.step(a)
        succ += int(info["success"]); dists.append(info["dist"])
    return succ / episodes, float(np.mean(dists))

def eval_feudal(env, hl, ll, K, episodes=50):
    succ = 0; dists = []
    for _ in range(episodes):
        o = env.reset(); done = False; steps = 0
        subgoal = None; sg_obs = None
        while not done:
            if steps % K == 0:
                a_hl = hl.act(o, noise=0.0)
                subgoal = env.subgoal_from_action(a_hl)
            lo = env.obs_for_subgoal(subgoal)
            a_ll = ll.act(lo, noise=0.0)
            o, r, done, info = env.step(a_ll)
            steps += 1
        succ += int(info["success"]); dists.append(info["dist"])
    return succ / episodes, float(np.mean(dists))

# ----------------------------- training loops -----------------------------
def train_flat(env, seed, total_steps, out_path, meta, start_random=5000,
               eval_every=20000, eval_eps=50, expl_noise=0.2, bs=256):
    obs_dim = 4; act_dim = 2
    agent = TD3(obs_dim, act_dim, seed)
    rb = Replay(obs_dim, act_dim)
    curve = []
    o = env.reset(); ep_ret = 0; ep_succ = 0; best = 0.0
    t0 = time.time()
    for step in range(1, total_steps + 1):
        if step < start_random:
            a = np.random.uniform(-1, 1, act_dim)
        else:
            a = agent.act(o, noise=expl_noise)
        o2, r, done, info = env.step(a)
        rb.add(o, a, r, o2, float(info["success"]))  # done-for-bootstrap only on true success
        o = o2; ep_ret += r
        if done:
            ep_succ = int(info["success"]); o = env.reset(); ep_ret = 0
        if step >= start_random and len(rb) >= bs:
            agent.update(rb.sample(bs))
        if step % eval_every == 0 or step == total_steps:
            sr, md = eval_flat(env, agent, eval_eps)
            best = max(best, sr)
            curve.append({"step": step, "success": sr, "mean_dist": md,
                          "elapsed": round(time.time() - t0, 1)})
            _dump(out_path, meta, curve, best, final=(step == total_steps))
            print(f"[flat s{seed} {meta['maze']}] step {step} succ {sr:.3f} best {best:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return best

def train_feudal(env, seed, total_steps, out_path, meta, K=15, start_random=5000,
                 eval_every=20000, eval_eps=50, expl_noise=0.2, bs=256):
    obs_dim = 4; act_dim = 2
    hl = TD3(obs_dim, act_dim, seed, gamma=0.98)          # HL obs=[x,y,gx,gy], act=subgoal
    ll = TD3(obs_dim, act_dim, seed + 12345, gamma=0.95)  # LL obs=[x,y,sgx,sgy], act=move
    rb_hl = Replay(obs_dim, act_dim); rb_ll = Replay(obs_dim, act_dim)
    curve = []; best = 0.0
    t0 = time.time()
    o = env.reset()
    step = 0
    next_eval = eval_every
    while step < total_steps:
        # --- HL decision ---
        if step < start_random:
            a_hl = np.random.uniform(-1, 1, act_dim)
        else:
            a_hl = hl.act(o, noise=expl_noise)
        subgoal = env.subgoal_from_action(a_hl)
        hl_o = o.copy()
        hl_rew = 0.0
        done = False
        prev_dist = env.dist_to(subgoal)
        for k in range(K):
            lo = env.obs_for_subgoal(subgoal)
            if step < start_random:
                a_ll = np.random.uniform(-1, 1, act_dim)
            else:
                a_ll = ll.act(lo, noise=expl_noise)
            o2, r_env, done, info = env.step(a_ll)
            step += 1
            hl_rew += r_env
            # intrinsic LL reward: progress toward subgoal + reach bonus (self-generated)
            cur_dist = env.dist_to(subgoal)
            reach = cur_dist <= env.sg_radius
            r_ll = (prev_dist - cur_dist) + (1.0 if reach else 0.0)
            prev_dist = cur_dist
            lo2 = env.obs_for_subgoal(subgoal)
            ll_done = 1.0 if reach else 0.0
            rb_ll.add(lo, a_ll, r_ll, lo2, ll_done)
            o = o2
            # LL update every env step
            if step >= start_random and len(rb_ll) >= bs:
                ll.update(rb_ll.sample(bs))
            # HL update at LL cadence too (uses its own buffer) -- keeps grad steps comparable
            if step >= start_random and len(rb_hl) >= bs:
                hl.update(rb_hl.sample(bs))
            if done or reach:
                break
            if step >= total_steps:
                break
        # --- HL transition (coarse timescale) ---
        hl_done = 1.0 if info["success"] else 0.0
        rb_hl.add(hl_o, a_hl, hl_rew, o, hl_done)
        if done:
            o = env.reset()
        if step >= next_eval or step >= total_steps:
            next_eval += eval_every
            sr, md = eval_feudal(env, hl, ll, K, eval_eps)
            best = max(best, sr)
            curve.append({"step": step, "success": sr, "mean_dist": md,
                          "elapsed": round(time.time() - t0, 1)})
            _dump(out_path, meta, curve, best, final=(step >= total_steps))
            print(f"[feudal s{seed} {meta['maze']}] step {step} succ {sr:.3f} best {best:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return best

def _dump(out_path, meta, curve, best, final=False):
    rec = dict(meta)
    rec["curve"] = curve
    rec["peak_success"] = best
    rec["final_success"] = curve[-1]["success"] if curve else 0.0
    rec["done"] = bool(final)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, out_path)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["flat", "feudal"], required=True)
    ap.add_argument("--maze", choices=["room", "midroom", "bigroom", "corridor", "umaze", "fourroom"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=800000)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    env = PointMaze(args.maze, seed=args.seed)
    meta = {
        "arm": args.arm, "maze": args.maze, "seed": args.seed,
        "total_steps": args.steps, "K": args.K,
        "max_ep_steps": env.max_steps,
        "start": env.start.tolist(), "goal": env.goal.tolist(),
        "tag": args.tag,
    }
    name = f"{args.arm}_{args.maze}_s{args.seed}"
    out_path = os.path.join(args.out_dir, name + ".json")
    print(f"START {name}: max_ep_steps={env.max_steps} start={env.start} goal={env.goal}", flush=True)
    if args.arm == "flat":
        train_flat(env, args.seed, args.steps, out_path, meta)
    else:
        train_feudal(env, args.seed, args.steps, out_path, meta, K=args.K)
    print(f"DONE {name}", flush=True)

if __name__ == "__main__":
    main()
