"""H-JEPA on a 2D point-maze (nav-control positive control).

Adapted from run_hjepa.py (PandaPickCube smoke test) — the DECISIVE control test:
does our faithful 2-level H-JEPA cross competence on an EASY-low-level,
LONG-HORIZON goal-reaching task, where the LL primitive (move-toward-subgoal) is
TRIVIAL and the only challenge is temporal abstraction / distant-goal credit?

What changed vs run_hjepa.py:
  * ENV: brax PandaPickCube -> JAX PointMaze (pointmaze_jax.PointMaze). obs=[pos,goal]
    (4-d), act=2-d velocity. Success = reached goal (metrics["at_goal"]). LL is
    trivially feasible by construction (sanity_maze.py: scripted router = 1.00).
  * EVAL: box_target/reached_box -> at_goal / -dist. Success = ever-at-goal in ep.
  * BASELINE: --flat runs a SINGLE-LEVEL control with the SAME backbone (encoder +
    LL Pi/Q), but NO learned HL subgoal. The LL conditions on (z_t, z_goal) where
    z_goal = enc(goal-augmented obs) and is trained DIRECTLY on env reward. This
    isolates "learned temporal-abstraction subgoal" (2-level) vs "direct
    goal-conditioned single-level" (flat). Same nets, same budget -> fair control.

ALL 4 H-JEPA FAITHFULNESS PROPERTIES KEPT for the 2-level run:
  1. NON-GENERATIVE (no decoder, no recon loss).
  2. Latent-predictive loss w/ EMA target + SimNorm + VICReg anti-collapse, logged.
  3. TWO LEVELS: HL emits a latent subgoal g every k steps; LL reaches it.
  4. HL = reactive SAC head over jumpy latents (latent CEM is the documented next step).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

HELIOS = Path("/root/helios-rl")
sys.path.insert(0, str(HELIOS / "src"))
sys.path.insert(0, "/root/helios-rl/hjepa_navctrl")
from pointmaze_jax import PointMaze, EPISODE_LEN, GOAL_RADIUS  # noqa: E402
from helios.algorithms.tdmpc2 import (  # noqa: E402
    simnorm, NormMLP, Pi, QEnsemble,
    two_hot, two_hot_inv, soft_ce, MultiEnvBuffer,
)


class Encoder(nn.Module):
    latent_dim: int
    hidden: tuple = (256, 256)
    V: int = 8
    @nn.compact
    def __call__(self, obs):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)


class JumpyPredictor(nn.Module):
    latent_dim: int
    hidden: tuple = (256, 256)
    V: int = 8
    @nn.compact
    def __call__(self, z, a_concat):
        x = jnp.concatenate([z, a_concat], -1)
        return simnorm(NormMLP(self.hidden, self.latent_dim)(x), self.V)


class HLPolicy(nn.Module):
    latent_dim: int
    hidden: tuple = (256, 256)
    V: int = 8
    raw: bool = False
    @nn.compact
    def __call__(self, z):
        h = NormMLP(self.hidden, self.latent_dim)(z)
        if self.raw:
            # RAW mode: emit an obs-space subgoal directly (no SimNorm simplex).
            return h
        return simnorm(h, self.V)


def latent_stats(Z: np.ndarray, V: int = 8) -> dict:
    Z = np.asarray(Z, np.float64)
    n, D = Z.shape
    std = Z.std(axis=0)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = ev.sum()
    eff_rank = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0
    G = V
    grp = Z.reshape(n, G, D // G)
    codes = grp.argmax(-1)
    ent, used = [], []
    for g in range(G):
        cnt = np.bincount(codes[:, g], minlength=D // G).astype(np.float64)
        p = cnt / cnt.sum(); p = p[p > 0]
        ent.append(float(-(p * np.log(p)).sum()))
        used.append(int((cnt > 0).sum()))
    max_ent = float(np.log(D // G))
    return {
        "z_std_mean": float(std.mean()), "z_std_min": float(std.min()),
        "z_eff_rank": eff_rank, "z_dim": int(D),
        "simnorm_code_entropy_mean": float(np.mean(ent)),
        "simnorm_code_entropy_frac": float(np.mean(ent) / max_ent) if max_ent > 0 else 0.0,
        "simnorm_codes_used_mean": float(np.mean(used)),
        "simnorm_codes_per_group": int(D // G),
        "n_eval_latents": int(n),
    }


def vicreg(z, gamma=0.05):
    B, D = z.shape
    std = jnp.sqrt(z.var(axis=0) + 1e-4)
    v = jnp.mean(jax.nn.relu(gamma - std))
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / (B - 1)
    off = cov - jnp.diag(jnp.diag(cov))
    c = jnp.sum(off ** 2) / D
    return v, c


def goal_obs(o):
    """Build a 'goal-only' obs whose position == goal (used for z_goal in flat mode
    and as a sanity target). obs layout = [pos(2), goal(2)] -> [goal, goal]."""
    g = o[..., 2:]
    return jnp.concatenate([g, g], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=400_000)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--V", type=int, default=4)
    ap.add_argument("--k", type=int, default=25, help="HL option horizon (env steps per subgoal)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--utd", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--eval_interval", type=int, default=40_000)
    ap.add_argument("--eval_eps", type=int, default=64)
    ap.add_argument("--vicreg", type=float, default=1.0)
    ap.add_argument("--vic_gamma", type=float, default=0.05)
    ap.add_argument("--viccov", type=float, default=0.1)
    ap.add_argument("--env_rew_w", type=float, default=1.0)
    ap.add_argument("--jepa_w", type=float, default=1.0, help="weight on the JEPA latent-"
                    "predictive loss in the shared-encoder objective")
    ap.add_argument("--flat", action="store_true", help="single-level baseline (no HL subgoal)")
    ap.add_argument("--raw_obs", action="store_true", help="identity encoder (z=obs): "
                    "controller acts on RAW observations, NO learned latent. Isolates "
                    "whether the controller/hierarchy works when the representation is "
                    "stable (vs the SimNorm-latent collapse failure mode).")
    ap.add_argument("--walls", default="c", choices=["c", "none"], help="maze layout")
    ap.add_argument("--reward", default="dense", choices=["dense", "sparse"])
    ap.add_argument("--goal_radius", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/helios-rl/exp/hjepa_navctrl")
    args = ap.parse_args()

    RAW = args.raw_obs
    MODE = ("flat" if args.flat else "hjepa2") + ("_raw" if RAW else "")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    LD, V, K = args.latent_dim, args.V, args.k

    env = PointMaze(walls=args.walls, reward=args.reward, goal_radius=args.goal_radius)
    obs_dim, act_dim = env.observation_size, env.action_size
    if RAW:
        LD = obs_dim                 # identity encoder: latent == observation
        args.jepa_w = 0.0            # no JEPA loss on an identity encoder
    N = args.num_envs
    print(f"[{MODE}] env=PointMaze obs={obs_dim} act={act_dim} N={N} latent={LD} k={K} seed={args.seed}", flush=True)

    _reset = jax.jit(env.reset)
    _step = jax.jit(env.step)

    enc = Encoder(LD, V=V)
    pred = JumpyPredictor(LD, V=V)
    hl = HLPolicy(LD, V=V, raw=RAW)
    ll_pi = Pi(act_dim)
    ll_q = QEnsemble()

    key, *ks = jax.random.split(key, 8)
    p_enc = enc.init(ks[0], jnp.zeros((1, obs_dim)))
    p_pred = pred.init(ks[1], jnp.zeros((1, LD)), jnp.zeros((1, K * act_dim)))
    p_hl = hl.init(ks[2], jnp.zeros((1, LD)))
    p_pi = ll_pi.init(ks[3], jnp.zeros((1, 2 * LD)))
    p_q = ll_q.init(ks[4], jnp.zeros((1, 2 * LD)), jnp.zeros((1, act_dim)))

    params = {"enc": p_enc, "pred": p_pred, "hl": p_hl, "pi": p_pi, "q": p_q}
    # EMA targets for enc (JEPA), q (TD) AND pi (target-policy -> decouples actor
    # from critic; fixes the "reach-then-diverge" Q-overestimation oscillation).
    target = {"enc": jax.tree.map(lambda x: x, p_enc),
              "q": jax.tree.map(lambda x: x, p_q),
              "pi": jax.tree.map(lambda x: x, p_pi)}

    tx = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(args.lr))
    opt_state = tx.init(params)

    def Enc(p_enc, o):
        # identity encoder when RAW (z == obs); else the SimNorm JEPA encoder.
        return o if RAW else enc.apply(p_enc, o)

    enc_j = jax.jit(lambda p, o: Enc(p["enc"], o))
    enc_tgt_j = jax.jit(lambda tgt, o: Enc(tgt["enc"], o))

    @jax.jit
    def hl_subgoal(p, z):
        return hl.apply(p["hl"], z)

    @jax.jit
    def ll_act(p, z, g, key, noise):
        zg = jnp.concatenate([z, g], -1)
        mean, _ = ll_pi.apply(p["pi"], zg)
        a = jnp.tanh(mean) + noise * jax.random.normal(key, mean.shape)
        return jnp.clip(a, -1, 1)

    @jax.jit
    def ll_act_det(p, z, g):
        mean, _ = ll_pi.apply(p["pi"], jnp.concatenate([z, g], -1))
        return jnp.tanh(mean)

    def jepa_loss(params, target, o_t, a_concat, o_tk):
        z_t = Enc(params["enc"], o_t)
        zhat = pred.apply(params["pred"], z_t, a_concat)
        z_tgt = jax.lax.stop_gradient(Enc(target["enc"], o_tk))
        l_pred = jnp.mean(jnp.sum((zhat - z_tgt) ** 2, -1))
        z_tk = Enc(params["enc"], o_tk)
        v1, c1 = vicreg(z_t, args.vic_gamma)
        v2, c2 = vicreg(z_tk, args.vic_gamma)
        v3, c3 = vicreg(zhat, args.vic_gamma)
        return (l_pred + args.vicreg * (v1+v2+v3) + args.viccov * (c1+c2+c3),
                (l_pred, v1+v2+v3, c1+c2+c3))

    def ll_loss(params, target, o, a, r_int, o2, g, done):
        # Let the value loss CO-TRAIN the encoder (TD-MPC2 style): the encoder is no
        # longer shaped by JEPA alone (which was drifting/collapsing and breaking the
        # controller). Task-relevant TD gradients keep the latent useful for control.
        z = Enc(params["enc"], o)
        z2 = jax.lax.stop_gradient(Enc(params["enc"], o2))
        zg = jnp.concatenate([z, g], -1)
        zg2 = jnp.concatenate([z2, g], -1)
        mean2, _ = ll_pi.apply(target["pi"], zg2)   # TARGET policy for TD action
        a2 = jnp.tanh(mean2)
        q2 = ll_q.apply(target["q"], zg2, a2)
        q2 = jnp.min(two_hot_inv(q2), axis=-1)                # (B,) min over the 2 heads
        td_tgt = jax.lax.stop_gradient(r_int + args.gamma * (1.0 - done) * q2)   # (B,)
        q = ll_q.apply(params["q"], zg, a)                   # (B,2,101)
        tgt_2hot = two_hot(td_tgt)[:, None, :]               # (B,1,101) -> broadcasts to (B,2,101)
        l_q = jnp.mean(soft_ce(q, tgt_2hot))                 # (B,2) -> scalar
        mean, _ = ll_pi.apply(params["pi"], zg)
        a_pi = jnp.tanh(mean)
        q_pi = jnp.min(two_hot_inv(ll_q.apply(jax.lax.stop_gradient(params["q"]), zg, a_pi)), axis=-1)
        l_pi = -jnp.mean(q_pi) + 1e-3 * jnp.mean(mean ** 2)
        return l_q + l_pi, (l_q, l_pi)

    def hl_loss(params, o, o_future, env_ret):
        z = jax.lax.stop_gradient(Enc(params["enc"], o))
        g = hl.apply(params["hl"], z)
        z_future = jax.lax.stop_gradient(Enc(params["enc"], o_future))
        w = jax.nn.sigmoid(env_ret)[:, None]
        return jnp.mean(w * jnp.sum((g - z_future) ** 2, -1))

    @jax.jit
    def update(params, target, opt_state, batch):
        (o_t, a_concat, o_tk, o, a, r_int, o2, g, done, o_hl, o_fut, env_ret) = batch

        def total(p):
            lj, (lp, lv, lc) = jepa_loss(p, target, o_t, a_concat, o_tk)
            ll, (lq, lpi) = ll_loss(p, target, o, a, r_int, o2, g, done)
            lh = hl_loss(p, o_hl, o_fut, env_ret)
            # flat baseline: NO HL (drop the HL hindsight loss; HL net is unused, the
            # subgoal in ll_loss is z_goal not hl(z) -> see sample_batch).
            lh = jnp.where(jnp.bool_(args.flat), 0.0, lh)
            return args.jepa_w * lj + ll + lh, (lp, lv, lc, lq, lpi, lh)

        (loss, aux), grads = jax.value_and_grad(total, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target = {
            "enc": jax.tree.map(lambda t, o_: args.ema*t + (1-args.ema)*o_, target["enc"], params["enc"]),
            "q":   jax.tree.map(lambda t, o_: args.ema*t + (1-args.ema)*o_, target["q"], params["q"]),
            "pi":  jax.tree.map(lambda t, o_: args.ema*t + (1-args.ema)*o_, target["pi"], params["pi"]),
        }
        return params, target, opt_state, loss, aux

    buf = MultiEnvBuffer(cap=max(args.total_steps // N + 2000, 20_000),
                         n_envs=N, obs_dim=obs_dim, act_dim=act_dim, seq_len=K + 2)
    dist_scale = np.float64(1.0)

    def goal_subgoal(o_arr):
        """z_goal for FLAT mode: encode the true goal as a latent subgoal."""
        return np.asarray(enc_tgt_j(target, jnp.asarray(np.asarray(goal_obs(jnp.asarray(o_arr))))))

    def sample_batch():
        nonlocal dist_scale
        s = buf.sample(args.batch, rng)
        if s is None:
            return None
        obs_s, act_s, rew_s, done_s = s
        o_t = obs_s[:, 0]
        a_concat = act_s[:, :K].reshape(args.batch, K * act_dim)
        o_tk = obs_s[:, K]
        o = obs_s[:, 0]; a = act_s[:, 0]; o2 = obs_s[:, 1]; done = done_s[:, 0]
        if args.flat:
            # FLAT: subgoal = encoded TRUE goal; LL reward = env reward directly
            # (single-level, goal-conditioned). No temporal abstraction.
            g = goal_subgoal(o)
            r_int = rew_s[:, 0]
        else:
            # 2-level H-JEPA: subgoal = future latent (k ahead), intrinsic reward =
            # subgoal-distance reduction (normalized) + env bonus.
            g = np.asarray(enc_tgt_j(target, jnp.asarray(o_tk)))
            z = np.asarray(enc_tgt_j(target, jnp.asarray(o)))
            z2 = np.asarray(enc_tgt_j(target, jnp.asarray(o2)))
            d1 = np.linalg.norm(z - g, axis=-1)
            d2 = np.linalg.norm(z2 - g, axis=-1)
            dist_scale = 0.99 * dist_scale + 0.01 * float(np.mean(d1) + 1e-6)
            r_int = (d1 - d2) / (dist_scale + 1e-6) + args.env_rew_w * rew_s[:, 0]
        o_hl = obs_s[:, 0]; o_fut = obs_s[:, K]
        env_ret = rew_s[:, :K].sum(axis=-1)
        return (jnp.asarray(o_t), jnp.asarray(a_concat), jnp.asarray(o_tk),
                jnp.asarray(o), jnp.asarray(a), jnp.asarray(r_int.astype(np.float32)),
                jnp.asarray(o2), jnp.asarray(g), jnp.asarray(done.astype(np.float32)),
                jnp.asarray(o_hl), jnp.asarray(o_fut), jnp.asarray(env_ret.astype(np.float32)))

    FLAT = bool(args.flat)

    def _eval_rollout(params, est0, T):
        z0 = enc_j(params, est0.obs)
        if FLAT:
            g0 = enc_j(params, goal_obs(est0.obs))
        else:
            g0 = hl_subgoal(params, z0)

        def body(carry, _t):
            est, g, age, alive, ag_max, mindist, ret_sum = carry
            o = est.obs
            z = enc_j(params, o)
            if FLAT:
                g = enc_j(params, goal_obs(o))     # always condition on true goal
            else:
                refresh = age >= K
                g = jnp.where(refresh[:, None], hl_subgoal(params, z), g)
                age = jnp.where(refresh, 0, age)
            a = ll_act_det(params, z, g)
            est = _step(est, a)
            ag = jnp.asarray(est.metrics["at_goal"]).reshape(-1)
            dist = jnp.asarray(est.info["dist"]).reshape(-1)
            r = est.reward.reshape(-1)
            ag_max = jnp.maximum(ag_max, jnp.where(alive > 0.5, ag, -jnp.inf))
            mindist = jnp.minimum(mindist, jnp.where(alive > 0.5, dist, jnp.inf))
            ret_sum = ret_sum + alive * r
            done = est.done.reshape(-1)
            alive = alive * (1.0 - (done > 0.5).astype(jnp.float32))
            age = age + 1
            return (est, g, age, alive, ag_max, mindist, ret_sum), z

        B = est0.obs.shape[0]
        carry0 = (est0, g0, jnp.zeros(B, jnp.int32), jnp.ones(B, jnp.float32),
                  jnp.full(B, -jnp.inf), jnp.full(B, jnp.inf), jnp.zeros(B, jnp.float32))
        (carry, Z) = jax.lax.scan(body, carry0, jnp.arange(T))
        _, _, _, _, ag_max, mindist, ret_sum = carry
        return ag_max, mindist, ret_sum, Z

    _eval_rollout_j = jax.jit(_eval_rollout, static_argnums=(2,))

    def evaluate(n_eps):
        nonlocal key
        key, rk2 = jax.random.split(key)
        est0 = _reset(jax.random.split(rk2, n_eps))
        ag_max, mindist, ret_sum, Z = _eval_rollout_j(params, est0, EPISODE_LEN)
        ag_max = np.asarray(ag_max); mindist = np.asarray(mindist); ret_sum = np.asarray(ret_sum)
        succ = (ag_max >= 0.5).astype(np.float64)
        Z = np.asarray(Z).reshape(-1, Z.shape[-1])
        if Z.shape[0] > 4000:
            Z = Z[rng.choice(Z.shape[0], 4000, replace=False)]
        ls = latent_stats(Z, V=V) if Z.shape[0] else {}
        return (float(np.mean(succ)), float(np.mean(ret_sum)), float(np.mean(mindist)),
                int(n_eps), ls)

    key, rk = jax.random.split(key)
    st = _reset(jax.random.split(rk, N))
    cur_g = None
    g_age = np.zeros(N, np.int32)

    metrics_log = []
    t0 = time.time()
    env_steps = 0; n_updates = 0; last_eval = 0
    aux = (0,)*6; loss = 0.0

    while env_steps < args.total_steps:
        z = enc_j(params, jnp.asarray(st.obs))
        if FLAT:
            cur_g = enc_j(params, goal_obs(jnp.asarray(st.obs)))
        else:
            if cur_g is None:
                cur_g = hl_subgoal(params, z)
            refresh = (g_age >= K)
            if refresh.any():
                new_g = hl_subgoal(params, z)
                cur_g = jnp.where(jnp.asarray(refresh)[:, None], new_g, cur_g)
                g_age[refresh] = 0
        key, ak = jax.random.split(key)
        noise = 0.3 if env_steps < args.warmup * 4 else 0.1
        if env_steps < args.warmup:
            key, rak = jax.random.split(key)
            a = jax.random.uniform(rak, (N, act_dim), minval=-1, maxval=1)
        else:
            a = ll_act(params, z, cur_g, ak, noise)
        a_np = np.asarray(a)
        nst = _step(st, a)
        r_np = np.asarray(nst.reward)
        done_np = np.asarray(nst.done).astype(np.float32)
        buf.add_batch(np.asarray(st.obs), a_np, r_np, done_np)
        st = nst
        g_age += 1
        env_steps += N

        if env_steps >= args.warmup and buf.total_size() > args.batch * 2:
            for _ in range(args.utd):
                batch = sample_batch()
                if batch is None:
                    break
                params, target, opt_state, loss, aux = update(params, target, opt_state, batch)
                n_updates += 1

        if env_steps - last_eval >= args.eval_interval or env_steps >= args.total_steps:
            last_eval = env_steps
            lp, lv, lc, lq, lpi, lh = [float(x) for x in aux] if n_updates > 0 else (0,)*6
            s_rate, s_ret, s_mindist, neps, ls = evaluate(args.eval_eps)
            rec = {"mode": MODE, "env_steps": int(env_steps), "n_updates": int(n_updates),
                   "wall_s": round(time.time()-t0, 1),
                   "loss_jepa_pred": lp, "loss_vicreg_var": lv, "loss_vicreg_cov": lc,
                   "loss_ll_q": lq, "loss_ll_pi": lpi, "loss_hl": lh,
                   "eval_success": s_rate, "eval_return": s_ret, "eval_min_dist": s_mindist,
                   "eval_n": neps, **ls}
            metrics_log.append(rec)
            print(f"[{MODE}] step={env_steps:>7} upd={n_updates:>6} "
                  f"Lpred={lp:.4f} Lvar={lv:.4f} Lq={lq:.3f} Lpi={lpi:.3f} Lhl={lh:.4f} | "
                  f"succ={s_rate:.3f} min_dist={s_mindist:.3f} ret={s_ret:.1f} | "
                  f"z_std={ls.get('z_std_mean',0):.4f} eff_rank={ls.get('z_eff_rank',0):.1f}/{LD} "
                  f"code_ent={ls.get('simnorm_code_entropy_frac',0):.2f} | {rec['wall_s']:.0f}s", flush=True)
            with open(outdir / f"metrics_{MODE}_seed{args.seed}.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    if metrics_log:
        first, last = metrics_log[0], metrics_log[-1]
        best_succ = max(m["eval_success"] for m in metrics_log)
        best_mindist = min(m["eval_min_dist"] for m in metrics_log)
        ce = last.get("simnorm_code_entropy_frac", 0.0)
        er = last.get("z_eff_rank", 0.0)
        collapsed = (ce < 0.05) or (er < 2.0) or (last.get("z_std_mean", 0) < 1e-3)
        summary = {
            "run": "hjepa_navctrl", "mode": MODE, "env": "PointMaze", "seed": args.seed,
            "flat_baseline": FLAT, "two_level": (not FLAT),
            "total_env_steps": int(env_steps), "n_updates": int(n_updates),
            "k_option_horizon": K, "latent_dim": LD, "V": V, "episode_len": EPISODE_LEN,
            "goal_radius": GOAL_RADIUS,
            "non_generative": True, "decoder": False, "ema_target_encoder": True,
            "anti_collapse": ["simnorm", "vicreg_var", "vicreg_cov"],
            "jepa_pred_loss_first": first["loss_jepa_pred"],
            "jepa_pred_loss_last": last["loss_jepa_pred"],
            "latent_collapsed": bool(collapsed),
            "z_eff_rank_last": er, "z_std_mean_last": last.get("z_std_mean", 0.0),
            "simnorm_code_entropy_frac_last": ce,
            "best_eval_success": float(best_succ),
            "best_eval_min_dist": float(best_mindist),
            "final_eval_success": last["eval_success"],
            "final_eval_min_dist": last["eval_min_dist"],
            "eval_n": last["eval_n"], "wall_s": last["wall_s"],
        }
        with open(outdir / f"summary_{MODE}_seed{args.seed}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{MODE}] SUMMARY " + json.dumps(summary), flush=True)
    print(f"[{MODE}] DONE", flush=True)


if __name__ == "__main__":
    main()
