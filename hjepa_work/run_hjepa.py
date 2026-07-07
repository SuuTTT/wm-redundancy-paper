"""H-JEPA: Hierarchical Joint-Embedding Predictive Architecture (smoke test).

A minimal-but-FAITHFUL H-JEPA on PandaPickCube, built on the helios-rl JAX infra
(env wrapper + SimNorm Encoder + MultiEnvBuffer + Pi/QEnsemble reused from tdmpc2).

FAITHFULNESS (the defining JEPA properties — none cut):
  1. NON-GENERATIVE. There is NO decoder and NO reconstruction loss anywhere.
     The world model predicts only in LATENT space.
  2. Latent-predictive loss with ANTI-COLLAPSE. enc(o)->z (SimNorm-bounded).
     The JEPA world-model / HL predictor pred(z_t, a_{t..t+k-1}) -> zhat_{t+k}
     is trained against  sg(target_enc(o_{t+k}))  where target_enc is an EMA copy
     of enc (BYOL/JEPA stop-gradient on the target). SimNorm gives built-in
     anti-collapse; we ALSO add a VICReg variance hinge on z and we LOG per-dim
     std + effective rank of z every eval to verify non-collapse empirically.
  3. TWO LEVELS (multi-timescale). HL operates on k-step-jumpy latents (k = option
     horizon ~150 env steps) and emits a LATENT subgoal g (a point in z-space).
     LL conditions on (z_t, g) and acts to reduce ||enc(o)-g||.
     LL reward = subgoal-distance reduction + small env-reward bonus.
  4. PLANNING at HL: this smoke test uses a REACTIVE HL policy emitting g (SAC head
     over jumpy latents). Latent-space CEM/MPPI rollout with the jumpy predictor is
     the documented next step (the predictor is already the model needed for it).

GPU0 only. Non-generative, EMA-target, 2-level. Reuses helios networks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# helios-rl infra
HELIOS = Path("/root/tdmpc_glass/helios-rl")
sys.path.insert(0, str(HELIOS / "src"))
sys.path.insert(0, "/root/tdmpc_glass/mujoco_playground_repo")
from mujoco_playground import registry, wrapper  # noqa: E402
from helios.algorithms.tdmpc2 import (  # noqa: E402
    simnorm, NormMLP, Pi, QEnsemble, sample_pi,
    two_hot, two_hot_inv, soft_ce, MultiEnvBuffer,
)


# ---------------------------------------------------------------------------
# Modules (all SimNorm-bounded latents; NO decoder anywhere)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    latent_dim: int
    hidden: tuple = (512, 512)
    V: int = 8
    @nn.compact
    def __call__(self, obs):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)


class JumpyPredictor(nn.Module):
    """JEPA world model: predict z_{t+k} from z_t and the k concatenated actions.
    Latent-only, SimNorm output. One call per k-step jump (multi-timescale HL)."""
    latent_dim: int
    hidden: tuple = (512, 512)
    V: int = 8
    @nn.compact
    def __call__(self, z, a_concat):
        x = jnp.concatenate([z, a_concat], -1)
        return simnorm(NormMLP(self.hidden, self.latent_dim)(x), self.V)


class HLPolicy(nn.Module):
    """High level: reactive policy emitting a LATENT subgoal g (a point in z-space,
    SimNorm-bounded so it lives on the same simplex manifold as encoder latents)."""
    latent_dim: int
    hidden: tuple = (512, 512)
    V: int = 8
    @nn.compact
    def __call__(self, z):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(z), self.V)


# ---------------------------------------------------------------------------
# Latent diagnostics (anti-collapse verification)
# ---------------------------------------------------------------------------
def latent_stats(Z: np.ndarray, V: int = 8) -> dict:
    """Anti-collapse diagnostics for the SimNorm latent.

    Reports (a) per-dim std, (b) PR effective rank of the raw covariance, and — the
    SimNorm-faithful one — (c) per-group CODE ENTROPY: each of V groups is a softmax
    over D//V dims; the argmax index is the "code". Collapse <=> every state maps to
    the same code in every group (mean entropy -> 0, n_used_codes -> 1). A healthy
    encoder uses many codes (high entropy). This is the definitive non-collapse signal
    for simplex latents (PR-eff-rank under-reads because each softmax concentrates mass).
    """
    Z = np.asarray(Z, np.float64)
    n, D = Z.shape
    std = Z.std(axis=0)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = ev.sum()
    eff_rank = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0
    # SimNorm code entropy
    G = V
    grp = Z.reshape(n, G, D // G)
    codes = grp.argmax(-1)                       # (n, G) code per group
    ent = []
    used = []
    for g in range(G):
        cnt = np.bincount(codes[:, g], minlength=D // G).astype(np.float64)
        p = cnt / cnt.sum()
        p = p[p > 0]
        ent.append(float(-(p * np.log(p)).sum()))
        used.append(int((cnt > 0).sum()))
    max_ent = float(np.log(D // G))
    return {
        "z_std_mean": float(std.mean()),
        "z_std_min": float(std.min()),
        "z_std_max": float(std.max()),
        "z_eff_rank": eff_rank,
        "z_dim": int(D),
        "simnorm_code_entropy_mean": float(np.mean(ent)),
        "simnorm_code_entropy_max": max_ent,
        "simnorm_code_entropy_frac": float(np.mean(ent) / max_ent) if max_ent > 0 else 0.0,
        "simnorm_codes_used_mean": float(np.mean(used)),
        "simnorm_codes_per_group": int(D // G),
        "n_eval_latents": int(n),
    }


def vicreg(z, gamma=1.0):
    """VICReg variance + covariance regularizer (anti-collapse).

    SimNorm latents live on V softmax simplices so per-dim values are in [0,1] and
    batch std is naturally small; gamma is calibrated accordingly (~0.05) and we ALSO
    add a covariance term that decorrelates latent dims — this is what actually breaks
    the "all states -> same simplex vertex" collapse that a variance hinge alone misses.
    Returns (variance_hinge, covariance) so they can be weighted separately and logged.
    """
    B, D = z.shape
    std = jnp.sqrt(z.var(axis=0) + 1e-4)
    v = jnp.mean(jax.nn.relu(gamma - std))            # variance hinge (per-dim spread)
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / (B - 1)
    off = cov - jnp.diag(jnp.diag(cov))
    c = jnp.sum(off ** 2) / D                          # covariance (decorrelation)
    return v, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PandaPickCube")
    ap.add_argument("--total_steps", type=int, default=300_000)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--k", type=int, default=150, help="HL option horizon (env steps per subgoal)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=4, help="LL TD sequence length")
    ap.add_argument("--ema", type=float, default=0.99, help="target encoder EMA decay")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--utd", type=int, default=1, help="gradient updates per env-collect step")
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--eval_interval", type=int, default=50_000)
    ap.add_argument("--eval_eps", type=int, default=64)
    ap.add_argument("--vicreg", type=float, default=3.0, help="VICReg var hinge weight "
                    "(raised 1.0->3.0: strengthen anti-collapse floor for the long run)")
    ap.add_argument("--vic_gamma", type=float, default=0.05,
                    help="VICReg target per-dim std (calibrated for SimNorm simplex latents)")
    ap.add_argument("--viccov", type=float, default=1.0, help="VICReg covariance weight")
    ap.add_argument("--env_rew_w", type=float, default=0.2, help="LL env-reward bonus weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/tdmpc_glass/exp/hjepa")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    LD, V, K = args.latent_dim, args.V, args.k

    # ── Env
    env = registry.load(args.env, config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    N = args.num_envs
    print(f"[hjepa] env={args.env} obs={obs_dim} act={act_dim} N={N} latent={LD} k={K}", flush=True)

    _reset = jax.jit(env.reset)
    _step = jax.jit(env.step)

    # ── Networks
    enc = Encoder(LD, V=V)
    pred = JumpyPredictor(LD, V=V)
    hl = HLPolicy(LD, V=V)
    ll_pi = Pi(act_dim)                     # LL actor: input (z|g)
    ll_q = QEnsemble()                      # LL critic: input (z|g), a

    key, *ks = jax.random.split(key, 8)
    p_enc = enc.init(ks[0], jnp.zeros((1, obs_dim)))
    p_pred = pred.init(ks[1], jnp.zeros((1, LD)), jnp.zeros((1, K * act_dim)))
    p_hl = hl.init(ks[2], jnp.zeros((1, LD)))
    p_pi = ll_pi.init(ks[3], jnp.zeros((1, 2 * LD)))
    p_q = ll_q.init(ks[4], jnp.zeros((1, 2 * LD)), jnp.zeros((1, act_dim)))

    params = {"enc": p_enc, "pred": p_pred, "hl": p_hl, "pi": p_pi, "q": p_q}
    target = {"enc": jax.tree.map(lambda x: x, p_enc),   # EMA target encoder (JEPA)
              "q": jax.tree.map(lambda x: x, p_q)}        # EMA target critic (TD)

    tx = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(args.lr))
    opt_state = tx.init(params)

    # ── apply fns
    def encode(p, o):
        return enc.apply(p["enc"], o)

    @jax.jit
    def hl_subgoal(p, z):
        return hl.apply(p["hl"], z)

    @jax.jit
    def ll_act(p, z, g, key, noise):
        zg = jnp.concatenate([z, g], -1)
        mean, log_std = ll_pi.apply(p["pi"], zg)
        a = jnp.tanh(mean)
        a = a + noise * jax.random.normal(key, a.shape)
        return jnp.clip(a, -1, 1)

    @jax.jit
    def ll_act_det(p, z, g):
        mean, _ = ll_pi.apply(p["pi"], jnp.concatenate([z, g], -1))
        return jnp.tanh(mean)

    # ── losses
    def jepa_loss(params, target, o_t, a_concat, o_tk):
        """Non-generative JEPA loss: predict EMA-target latent of o_{t+k}."""
        z_t = enc.apply(params["enc"], o_t)
        zhat = pred.apply(params["pred"], z_t, a_concat)
        z_tgt = enc.apply(target["enc"], o_tk)          # EMA target encoder...
        z_tgt = jax.lax.stop_gradient(z_tgt)            # ...stop-gradient (BYOL/JEPA)
        # cosine-ish + MSE in simplex space
        l_pred = jnp.mean(jnp.sum((zhat - z_tgt) ** 2, -1))
        # anti-collapse: VICReg variance + covariance on the ONLINE latents (z_t, z_{t+k})
        # and on the PREDICTOR output (so the predictor can't collapse to a constant).
        z_tk = enc.apply(params["enc"], o_tk)
        v1, c1 = vicreg(z_t, gamma=args.vic_gamma)
        v2, c2 = vicreg(z_tk, gamma=args.vic_gamma)
        v3, c3 = vicreg(zhat, gamma=args.vic_gamma)
        l_var = v1 + v2 + v3
        l_cov = c1 + c2 + c3
        return (l_pred + args.vicreg * l_var + args.viccov * l_cov,
                (l_pred, l_var, l_cov))

    def ll_loss(params, target, o, a, r_int, o2, g, done):
        """LL SAC-style update on subgoal-distance reward (latent space)."""
        z = jax.lax.stop_gradient(enc.apply(params["enc"], o))   # LL uses frozen enc latents
        z2 = jax.lax.stop_gradient(enc.apply(params["enc"], o2))
        zg = jnp.concatenate([z, g], -1)
        zg2 = jnp.concatenate([z2, g], -1)
        # target Q
        mean2, log_std2 = ll_pi.apply(params["pi"], zg2)
        a2 = jnp.tanh(mean2)
        q2 = ll_q.apply(target["q"], zg2, a2)             # (...,2,bins)
        q2 = jnp.min(two_hot_inv(q2), axis=-1, keepdims=True)
        td_tgt = r_int[:, None] + args.gamma * (1.0 - done[:, None]) * q2
        td_tgt = jax.lax.stop_gradient(td_tgt)
        q = ll_q.apply(params["q"], zg, a)
        tgt_2hot = two_hot(td_tgt)[:, None, :]
        l_q = jnp.mean(soft_ce(q, tgt_2hot))
        # actor: maximize Q (reparam)
        mean, log_std = ll_pi.apply(params["pi"], zg)
        a_pi = jnp.tanh(mean)
        q_pi = ll_q.apply(jax.lax.stop_gradient(params["q"]), zg, a_pi)
        q_pi = jnp.min(two_hot_inv(q_pi), axis=-1)
        l_pi = -jnp.mean(q_pi) + 1e-3 * jnp.mean(mean ** 2)
        return l_q + l_pi, (l_q, l_pi)

    def hl_loss(params, o, g_target_unused, o_future, env_ret):
        """HL reactive policy: emit subgoal g that points toward HIGH-env-return
        future latents. Smoke-test surrogate: regress g toward the EMA-target latent
        of a future state reached under good return (self-supervised hindsight goal).
        This keeps HL non-generative (g lives in z-space) and reactive."""
        z = jax.lax.stop_gradient(enc.apply(params["enc"], o))
        g = hl.apply(params["hl"], z)
        z_future = jax.lax.stop_gradient(enc.apply(params["enc"], o_future))
        # hindsight: subgoal should match the achieved future latent (weighted by return)
        w = jax.nn.sigmoid(env_ret)[:, None]
        return jnp.mean(w * jnp.sum((g - z_future) ** 2, -1))

    @jax.jit
    def update(params, target, opt_state, batch):
        (o_t, a_concat, o_tk,            # JEPA
         o, a, r_int, o2, g, done,       # LL
         o_hl, o_fut, env_ret) = batch

        def total(p):
            lj, (lp, lv, lc) = jepa_loss(p, target, o_t, a_concat, o_tk)
            ll, (lq, lpi) = ll_loss(p, target, o, a, r_int, o2, g, done)
            lh = hl_loss(p, o_hl, None, o_fut, env_ret)
            return lj + ll + lh, (lp, lv, lc, lq, lpi, lh)

        (loss, aux), grads = jax.value_and_grad(total, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # EMA targets
        target = {
            "enc": jax.tree.map(lambda t, o_: args.ema * t + (1 - args.ema) * o_,
                                target["enc"], params["enc"]),
            "q": jax.tree.map(lambda t, o_: args.ema * t + (1 - args.ema) * o_,
                              target["q"], params["q"]),
        }
        return params, target, opt_state, loss, aux

    # ── Replay buffer (reuse helios MultiEnvBuffer; seq long enough for k-jump)
    buf = MultiEnvBuffer(cap=max(args.total_steps // N + 2000, 20_000),
                         n_envs=N, obs_dim=obs_dim, act_dim=act_dim, seq_len=K + 2)

    enc_j = jax.jit(lambda p, o: enc.apply(p["enc"], o))
    # FROZEN/EMA-target encoder for LL reward (reward-decoupling fix): the subgoal
    # and the subgoal-distance reward are computed with the EMA-TARGET encoder, NOT
    # the online encoder that jepa_loss shapes. This stops the LL from earning reward
    # by shrinking the online latent manifold (the diagnosed back-half contraction).
    enc_tgt_j = jax.jit(lambda tgt, o: enc.apply(tgt["enc"], o))
    # running scale of subgoal distance (so a globally-contracting manifold earns no
    # free reward): we normalize (d1 - d2) by an EMA of the typical |z-g| magnitude.
    dist_scale = np.float64(1.0)

    # ── Rollout state
    key, rk = jax.random.split(key)
    st = _reset(jax.random.split(rk, N))
    cur_g = None
    g_age = np.zeros(N, np.int32)
    ep_ret = np.zeros(N, np.float32)

    metrics_log = []
    t0 = time.time()
    env_steps = 0
    n_updates = 0
    last_eval = 0

    def sample_batch():
        nonlocal dist_scale
        # JEPA k-jump samples: (o_t, a_{t:t+k}, o_{t+k})
        s = buf.sample(args.batch, rng)
        if s is None:
            return None
        obs_s, act_s, rew_s, done_s = s            # (B, K+2, *)
        o_t = obs_s[:, 0]
        a_concat = act_s[:, :K].reshape(args.batch, K * act_dim)
        o_tk = obs_s[:, K]
        # LL 1-step samples (from start of seq), hindsight subgoal = latent K ahead
        o = obs_s[:, 0]; a = act_s[:, 0]; o2 = obs_s[:, 1]; done = done_s[:, 0]
        # REWARD-DECOUPLING FIX: subgoal g AND the latents used for the LL reward are
        # encoded with the FROZEN EMA-target encoder, so the LL reward cannot back the
        # online encoder into contracting the manifold. (The subgoal g passed to ll_loss
        # must match what the LL is trained to reach; ll_loss stop-grads the online enc,
        # but the reward distances are now defined purely by the slow target encoder.)
        g = np.asarray(enc_tgt_j(target, jnp.asarray(o_tk)))   # subgoal = future latent (frozen enc)
        z = np.asarray(enc_tgt_j(target, jnp.asarray(o)))
        z2 = np.asarray(enc_tgt_j(target, jnp.asarray(o2)))
        # intrinsic LL reward = reduction in ||z - g|| + env bonus, NORMALIZED by a
        # running scale of the typical distance so a globally-shrinking manifold earns
        # no free reward (only relative progress toward the goal is rewarded).
        d1 = np.linalg.norm(z - g, axis=-1)
        d2 = np.linalg.norm(z2 - g, axis=-1)
        dist_scale = 0.99 * dist_scale + 0.01 * float(np.mean(d1) + 1e-6)
        r_int = (d1 - d2) / (dist_scale + 1e-6) + args.env_rew_w * rew_s[:, 0]
        # HL hindsight
        o_hl = obs_s[:, 0]; o_fut = obs_s[:, K]
        env_ret = rew_s[:, :K].sum(axis=-1)
        return (jnp.asarray(o_t), jnp.asarray(a_concat), jnp.asarray(o_tk),
                jnp.asarray(o), jnp.asarray(a), jnp.asarray(r_int.astype(np.float32)),
                jnp.asarray(o2), jnp.asarray(g), jnp.asarray(done.astype(np.float32)),
                jnp.asarray(o_hl), jnp.asarray(o_fut),
                jnp.asarray(env_ret.astype(np.float32)))

    # JITTED + VMAPPED eval: roll out all n_eps episodes in parallel as one batched
    # env via lax.scan (no Python step loop). HL subgoal refreshes every K steps; we
    # mask post-first-done steps per env so "ever succeeded/reached/return" matches the
    # original first-episode semantics. This removes the un-jitted single-env bottleneck.
    def _eval_rollout(params, est0, T):
        z0 = enc_j(params, est0.obs)
        g0 = hl_subgoal(params, z0)

        def body(carry, _t):
            est, g, age, alive, bt_max, rc_max, ret_sum = carry
            o = est.obs
            z = enc_j(params, o)
            refresh = age >= K
            g = jnp.where(refresh[:, None], hl_subgoal(params, z), g)
            age = jnp.where(refresh, 0, age)
            a = ll_act_det(params, z, g)
            est = _step(est, a)
            bt = jnp.asarray(est.metrics["box_target"]).reshape(-1)
            rc = jnp.asarray(est.info["reached_box"]).reshape(-1)
            r = est.reward.reshape(-1)
            bt_max = jnp.maximum(bt_max, jnp.where(alive > 0.5, bt, -jnp.inf))
            rc_max = jnp.maximum(rc_max, jnp.where(alive > 0.5, rc, -jnp.inf))
            ret_sum = ret_sum + alive * r
            done = est.done.reshape(-1)
            alive = alive * (1.0 - (done > 0.5).astype(jnp.float32))
            age = age + 1
            return (est, g, age, alive, bt_max, rc_max, ret_sum), z

        B = est0.obs.shape[0]
        carry0 = (est0, g0, jnp.zeros(B, jnp.int32), jnp.ones(B, jnp.float32),
                  jnp.full(B, -jnp.inf), jnp.full(B, -jnp.inf), jnp.zeros(B, jnp.float32))
        (carry, Z) = jax.lax.scan(body, carry0, jnp.arange(T))
        _, _, _, _, bt_max, rc_max, ret_sum = carry
        return bt_max, rc_max, ret_sum, Z   # Z: (T, B, LD)

    _eval_rollout_j = jax.jit(_eval_rollout, static_argnums=(2,))

    def evaluate(n_eps):
        nonlocal key
        key, rk2 = jax.random.split(key)
        est0 = _reset(jax.random.split(rk2, n_eps))
        bt_max, rc_max, ret_sum, Z = _eval_rollout_j(params, est0, 1000)
        bt_max = np.asarray(bt_max); rc_max = np.asarray(rc_max); ret_sum = np.asarray(ret_sum)
        succ = (bt_max >= 0.9).astype(np.float64)
        reached = np.clip(rc_max, 0.0, None)
        # subsample latents for the collapse diagnostic (cap ~4000 like before)
        Z = np.asarray(Z).reshape(-1, Z.shape[-1])
        if Z.shape[0] > 4000:
            idx = rng.choice(Z.shape[0], 4000, replace=False)
            Z = Z[idx]
        ls = latent_stats(Z, V=V) if Z.shape[0] else {}
        return (float(np.mean(succ)), float(np.mean(ret_sum)), float(np.mean(reached)),
                int(n_eps), ls)

    # ── Main loop
    while env_steps < args.total_steps:
        z = enc_j(params, jnp.asarray(st.obs))
        # HL: refresh subgoal every K env-steps per env
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
        ep_ret += r_np
        ep_ret[done_np > 0.5] = 0.0

        # ── Train
        if env_steps >= args.warmup and buf.total_size() > args.batch * 2:
            for _ in range(args.utd):
                batch = sample_batch()
                if batch is None:
                    break
                params, target, opt_state, loss, aux = update(params, target, opt_state, batch)
                n_updates += 1

        # ── Eval
        if env_steps - last_eval >= args.eval_interval or env_steps >= args.total_steps:
            last_eval = env_steps
            lp, lv, lc, lq, lpi, lh = [float(x) for x in aux] if n_updates > 0 else (0,)*6
            s_rate, s_ret, s_reach, neps, ls = evaluate(args.eval_eps)
            rec = {
                "env_steps": int(env_steps), "n_updates": int(n_updates),
                "wall_s": round(time.time() - t0, 1),
                "loss_total": float(loss) if n_updates > 0 else None,
                "loss_jepa_pred": lp, "loss_vicreg_var": lv, "loss_vicreg_cov": lc,
                "loss_ll_q": lq, "loss_ll_pi": lpi, "loss_hl": lh,
                "eval_success": s_rate, "eval_return": s_ret,
                "eval_reached": s_reach, "eval_n": neps,
                **ls,
            }
            metrics_log.append(rec)
            print(f"[hjepa] step={env_steps:>7} upd={n_updates:>6} "
                  f"Lpred={lp:.4f} Lvar={lv:.4f} Lcov={lc:.4f} Lq={lq:.3f} Lpi={lpi:.3f} Lhl={lh:.4f} | "
                  f"succ={s_rate:.3f} reached={s_reach:.3f} ret={s_ret:.1f} | "
                  f"z_std={ls.get('z_std_mean',0):.4f} eff_rank={ls.get('z_eff_rank',0):.1f}/{LD} "
                  f"code_ent={ls.get('simnorm_code_entropy_frac',0):.2f} "
                  f"codes_used={ls.get('simnorm_codes_used_mean',0):.1f}/{ls.get('simnorm_codes_per_group',0)} "
                  f"| {rec['wall_s']:.0f}s", flush=True)
            with open(outdir / f"metrics_seed{args.seed}.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    # ── Final smoke summary (self-contained deliverable JSON)
    if metrics_log:
        first, last = metrics_log[0], metrics_log[-1]
        best_succ = max(m["eval_success"] for m in metrics_log)
        best_reach = max(m["eval_reached"] for m in metrics_log)
        # non-collapse verdict: code entropy stayed well above 0 and eff_rank > a few
        ce = last.get("simnorm_code_entropy_frac", 0.0)
        er = last.get("z_eff_rank", 0.0)
        collapsed = (ce < 0.05) or (er < 2.0) or (last.get("z_std_mean", 0) < 1e-3)
        pred_dropped = (last["loss_jepa_pred"] < first["loss_jepa_pred"] * 0.9)
        summary = {
            "run": "hjepa_smoke", "env": args.env, "seed": args.seed,
            "total_env_steps": int(env_steps), "n_updates": int(n_updates),
            "k_option_horizon": K, "latent_dim": LD, "V": V,
            "non_generative": True, "decoder": False,
            "ema_target_encoder": True, "ema_decay": args.ema,
            "anti_collapse": ["simnorm", "vicreg_var", "vicreg_cov"],
            "jepa_pred_loss_first": first["loss_jepa_pred"],
            "jepa_pred_loss_last": last["loss_jepa_pred"],
            "jepa_pred_decreasing": bool(pred_dropped),
            "latent_collapsed": bool(collapsed),
            "z_eff_rank_last": er,
            "z_std_mean_last": last.get("z_std_mean", 0.0),
            "simnorm_code_entropy_frac_last": ce,
            "simnorm_codes_used_mean_last": last.get("simnorm_codes_used_mean", 0.0),
            "best_eval_success": float(best_succ),
            "best_eval_reached": float(best_reach),
            "final_eval_success": last["eval_success"],
            "final_eval_return": last["eval_return"],
            "eval_n": last["eval_n"],
            "wall_s": last["wall_s"],
        }
        with open(outdir / f"smoke_summary_seed{args.seed}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[hjepa] SUMMARY " + json.dumps(summary), flush=True)
    print("[hjepa] DONE", flush=True)


if __name__ == "__main__":
    main()
